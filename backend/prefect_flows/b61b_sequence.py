from __future__ import annotations

import copy
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from prefect_flows.b61b_core import (
    HAZARD_TARGETS,
    RANDOM_SEED,
    RISK_TARGETS,
    enforce_hazard_order,
    enforce_quantile_order,
    enforce_risk_order,
)


@dataclass
class SequenceResult:
    predictions: dict[str, np.ndarray]
    artifact: bytes
    metrics: dict[str, Any]


class LandmarkSequenceDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        group_starts: np.ndarray,
        endpoints: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray,
        weights: np.ndarray,
        sequence_length: int,
    ) -> None:
        self.values = values
        self.group_starts = group_starts
        self.endpoints = endpoints.astype("int64")
        self.targets = targets.astype("float32")
        self.masks = masks.astype("float32")
        self.weights = weights.astype("float32")
        self.sequence_length = int(sequence_length)

    def __len__(self) -> int:
        return len(self.endpoints)

    def __getitem__(self, item: int):
        endpoint = int(self.endpoints[item])
        start = max(int(self.group_starts[endpoint]), endpoint - self.sequence_length + 1)
        sequence = self.values[start : endpoint + 1]
        padded = np.zeros((self.sequence_length, self.values.shape[1]), dtype="float32")
        padded[-len(sequence) :] = sequence
        return (
            torch.from_numpy(padded),
            torch.tensor(len(sequence), dtype=torch.long),
            torch.from_numpy(self.targets[endpoint]),
            torch.from_numpy(self.masks[endpoint]),
            torch.tensor(self.weights[endpoint], dtype=torch.float32),
            torch.tensor(endpoint, dtype=torch.long),
        )


class MultiTaskGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 48) -> None:
        super().__init__()
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.risk_head = nn.Linear(hidden_size, 3)
        self.quantile_head = nn.Linear(hidden_size, 3)
        self.hazard_head = nn.Linear(hidden_size, 3)

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor):
        packed = nn.utils.rnn.pack_padded_sequence(
            sequence,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.encoder(packed)
        representation = self.shared(self.norm(hidden[-1]))
        risk = self.risk_head(representation)
        raw_quantiles = self.quantile_head(representation)
        p10 = torch.nn.functional.softplus(raw_quantiles[:, 0])
        p50 = p10 + torch.nn.functional.softplus(raw_quantiles[:, 1])
        p90 = p50 + torch.nn.functional.softplus(raw_quantiles[:, 2])
        return risk, torch.stack([p10, p50, p90], dim=1), self.hazard_head(representation)


def _pinball(actual: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    quantiles = torch.tensor([0.1, 0.5, 0.9], device=actual.device)
    residual = actual[:, None] - predicted
    return torch.maximum(quantiles * residual, (quantiles - 1.0) * residual).mean(dim=1)


def _loss(
    outputs,
    targets: torch.Tensor,
    masks: torch.Tensor,
    weights: torch.Tensor,
    positive_weights: torch.Tensor,
) -> torch.Tensor:
    risk_logits, quantiles, hazard_logits = outputs
    risk_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        risk_logits,
        targets[:, :3],
        reduction="none",
        pos_weight=positive_weights[:3],
    ).mean(dim=1)
    duration_loss = _pinball(targets[:, 3], quantiles)
    hazard_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        hazard_logits,
        targets[:, 4:7],
        reduction="none",
        pos_weight=positive_weights[3:],
    ).mean(dim=1)
    per_row = (
        risk_loss * masks[:, 0]
        + 0.35 * duration_loss * masks[:, 1]
        + 0.70 * hazard_loss * masks[:, 2]
    ) / torch.clamp(masks.sum(dim=1), min=1.0)
    return (per_row * weights).sum() / torch.clamp(weights.sum(), min=1.0)


def _prepare_arrays(frame: pd.DataFrame, features: list[str]):
    source = frame.sort_values(["port_call_id", "landmark_at"]).reset_index().rename(columns={"index": "source_index"})
    numeric = source[features].apply(pd.to_numeric, errors="coerce")
    train_mask = source["model_role"].eq("TRAIN_FIT")
    medians = numeric.loc[train_mask].median().fillna(0.0)
    numeric = numeric.fillna(medians)
    means = numeric.loc[train_mask].mean()
    scales = numeric.loc[train_mask].std().replace(0.0, 1.0).fillna(1.0)
    values = ((numeric - means) / scales).clip(-12.0, 12.0).to_numpy(dtype="float32")
    first = source.groupby("port_call_id", sort=False).cumcount().eq(0).to_numpy()
    first_positions = np.maximum.accumulate(np.where(first, np.arange(len(source)), 0))
    targets = np.column_stack(
        [
            *[source[column].astype("float32").to_numpy() for column in RISK_TARGETS],
            source["target_remaining_h"].astype("float32").to_numpy(),
            *[source[column].astype("float32").to_numpy() for column in HAZARD_TARGETS],
        ]
    ).astype("float32")
    targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
    masks = np.column_stack(
        [
            source["early_warning_eligible"].astype("float32"),
            source["target_remaining_h"].notna().astype("float32"),
            source["pre_breach_eligible"].astype("float32"),
        ]
    ).astype("float32")
    weights = pd.to_numeric(source["per_call_sample_weight"], errors="coerce").fillna(1.0).clip(0.01, 1.0).to_numpy(dtype="float32")
    return source, values, first_positions, targets, masks, weights, medians, means, scales


def train_sequence_expert(
    frame: pd.DataFrame,
    features: list[str],
    sequence_length: int = 24,
    max_steps: int = 400,
    batch_size: int = 256,
) -> SequenceResult:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.set_num_threads(2)
    source, values, group_starts, targets, masks, weights, medians, means, scales = _prepare_arrays(frame, features)
    train_endpoints = np.flatnonzero(source["model_role"].eq("TRAIN_FIT").to_numpy())
    valid_endpoints = np.flatnonzero(source["model_role"].eq("VALID_SELECT").to_numpy())
    all_endpoints = np.arange(len(source), dtype="int64")
    if len(train_endpoints) == 0 or len(valid_endpoints) == 0:
        raise ValueError("TRAIN_FIT and VALID_SELECT are required for the sequence expert")
    train_dataset = LandmarkSequenceDataset(values, group_starts, train_endpoints, targets, masks, weights, sequence_length)
    valid_dataset = LandmarkSequenceDataset(values, group_starts, valid_endpoints, targets, masks, weights, sequence_length)
    all_dataset = LandmarkSequenceDataset(values, group_starts, all_endpoints, targets, masks, weights, sequence_length)
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    all_loader = DataLoader(all_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    train_targets = targets[train_endpoints]
    train_masks = masks[train_endpoints]
    classification_targets = train_targets[:, [0, 1, 2, 4, 5, 6]]
    classification_masks = np.column_stack(
        [
            np.repeat(train_masks[:, [0]], 3, axis=1),
            np.repeat(train_masks[:, [2]], 3, axis=1),
        ]
    )
    positives = (classification_targets * classification_masks).sum(axis=0)
    negatives = classification_masks.sum(axis=0) - positives
    positive_weights = torch.tensor(np.clip(negatives / np.maximum(positives, 1.0), 0.5, 12.0), dtype=torch.float32)
    model = MultiTaskGRU(len(features))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = copy.deepcopy(model.state_dict())
    best_valid = float("inf")
    patience = 0
    steps = 0
    history = []
    for epoch in range(20):
        model.train()
        train_losses = []
        for sequence, lengths, target, mask, weight, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model(sequence, lengths), target, mask, weight, positive_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
            steps += 1
            if steps >= max_steps:
                break
        model.eval()
        valid_losses = []
        with torch.no_grad():
            for sequence, lengths, target, mask, weight, _ in valid_loader:
                valid_losses.append(float(_loss(model(sequence, lengths), target, mask, weight, positive_weights)))
        valid_loss = float(np.mean(valid_losses))
        history.append({"epoch": epoch + 1, "steps": steps, "train_loss": float(np.mean(train_losses)), "valid_loss": valid_loss})
        if valid_loss < best_valid - 1e-4:
            best_valid = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if steps >= max_steps or patience >= 3:
            break
    model.load_state_dict(best_state)
    model.eval()
    risk = np.zeros((len(source), 3), dtype="float32")
    quantiles = np.zeros((len(source), 3), dtype="float32")
    hazard = np.zeros((len(source), 3), dtype="float32")
    with torch.no_grad():
        for sequence, lengths, _, _, _, endpoints in all_loader:
            risk_logits, batch_quantiles, hazard_logits = model(sequence, lengths)
            positions = endpoints.numpy()
            risk[positions] = torch.sigmoid(risk_logits).numpy()
            quantiles[positions] = batch_quantiles.numpy()
            hazard[positions] = torch.sigmoid(hazard_logits).numpy()
    inverse = np.argsort(source["source_index"].to_numpy())
    risk = enforce_risk_order(risk)[inverse]
    quantiles = enforce_quantile_order(quantiles)[inverse]
    hazard = enforce_hazard_order(hazard)[inverse]
    buffer = io.BytesIO()
    torch.save(
        {
            "model_version": "b61b-gru-shared-encoder-v1",
            "state_dict": best_state,
            "features": features,
            "sequence_length": sequence_length,
            "hidden_size": 48,
            "medians": medians.to_dict(),
            "means": means.to_dict(),
            "scales": scales.to_dict(),
        },
        buffer,
    )
    return SequenceResult(
        predictions={"risk": risk, "quantiles": quantiles, "hazard": hazard},
        artifact=buffer.getvalue(),
        metrics={
            "train_rows": len(train_endpoints),
            "valid_select_rows": len(valid_endpoints),
            "features": len(features),
            "sequence_length": sequence_length,
            "steps": steps,
            "best_valid_loss": best_valid,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "history": history,
        },
    )
