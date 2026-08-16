FROM python:3.12-slim

ARG MLFLOW_VERSION=3.13.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir \
    "mlflow==${MLFLOW_VERSION}" \
    "psycopg2-binary==2.9.10" \
    "boto3>=1.35,<2"

COPY start-mlflow.sh /usr/local/bin/start-mlflow.sh
RUN chmod +x /usr/local/bin/start-mlflow.sh

EXPOSE 5000

CMD ["/usr/local/bin/start-mlflow.sh"]
