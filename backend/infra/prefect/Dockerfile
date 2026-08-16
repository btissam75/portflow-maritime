# syntax=docker/dockerfile:1.7
FROM prefecthq/prefect:3.8.0-python3.12

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=1200 \
    PIP_RETRIES=20 \
    PYTHONPATH=/opt/smart-port

WORKDIR /opt/smart-port

COPY infra/prefect/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 1200 --retries 20 -r /tmp/requirements.txt

COPY prefect_flows /opt/smart-port/prefect_flows
COPY prefect.yaml /opt/smart-port/prefect.yaml

CMD ["prefect", "worker", "start", "--pool", "smart-port-process", "--type", "process"]
