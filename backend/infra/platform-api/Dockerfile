# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/smart-port

COPY services/platform_api/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 1200 --retries 20 -r /tmp/requirements.txt

COPY services/platform_api /opt/smart-port/platform_api

EXPOSE 8092

CMD ["uvicorn", "platform_api.main:app", "--host", "0.0.0.0", "--port", "8092"]
