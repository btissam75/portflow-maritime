# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=1200 \
    PIP_RETRIES=20 \
    PYTHONPATH=/opt/smart-port

WORKDIR /opt/smart-port

COPY infra/model-trainer/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 1200 --retries 20 "catboost>=1.2.8,<2"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 1200 --retries 20 -r /tmp/requirements.txt

COPY services/model_trainer /opt/smart-port/model_trainer

EXPOSE 8091

CMD ["uvicorn", "model_trainer.main:app", "--host", "0.0.0.0", "--port", "8091"]
