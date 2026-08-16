FROM golang:1.24.8-alpine AS builder

ARG MINIO_VERSION=RELEASE.2025-10-15T17-29-55Z

RUN apk add --no-cache git
RUN CGO_ENABLED=0 GOBIN=/out go install -v "github.com/minio/minio@${MINIO_VERSION}"

FROM alpine:3.22

RUN apk add --no-cache ca-certificates curl tzdata \
    && addgroup -S minio \
    && adduser -S -G minio -h /home/minio minio \
    && mkdir -p /data \
    && chown -R minio:minio /data /home/minio

COPY --from=builder /out/minio /usr/local/bin/minio

USER minio
EXPOSE 9000 9001

ENTRYPOINT ["minio"]
CMD ["server", "/data", "--console-address", ":9001"]
