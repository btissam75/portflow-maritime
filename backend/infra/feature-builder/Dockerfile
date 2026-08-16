FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /opt/smart-port

COPY infra/feature-builder/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY services/feature_builder /opt/smart-port/feature_builder

EXPOSE 8090

CMD ["uvicorn", "feature_builder.main:app", "--host", "0.0.0.0", "--port", "8090", "--workers", "1"]
