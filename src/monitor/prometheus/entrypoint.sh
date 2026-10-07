#!/bin/sh
set -eu

TEMPLATE_PATH="/etc/prometheus/prometheus.yml"
OUTPUT_PATH="/tmp/prometheus.yml"

VLLM_TARGET="${VLLM_HOST:-127.0.0.1}:${VLLM_PORT:-8000}"
BACKEND_TARGET="${BACKEND_HOST:-127.0.0.1}:${BACKEND_PORT:-8001}"

sed \
  -e "s|__VLLM_TARGET__|${VLLM_TARGET}|g" \
  -e "s|__BACKEND_TARGET__|${BACKEND_TARGET}|g" \
  "$TEMPLATE_PATH" > "$OUTPUT_PATH"

exec /bin/prometheus --config.file="$OUTPUT_PATH"