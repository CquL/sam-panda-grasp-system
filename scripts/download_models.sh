#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/src/sam_perception/models"
MODEL_FILE="$MODEL_DIR/sam_vit_b_01ec64.pth"
MODEL_URL="${SAM_MODEL_URL:-https://github.com/CquL/sam-panda-grasp-system/releases/download/sam-models-v1/sam_vit_b_01ec64.pth}"
EXPECTED_SHA256="${SAM_MODEL_SHA256:-ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912}"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_FILE" ]; then
  echo "[OK] SAM checkpoint already exists: $MODEL_FILE"
else
  echo "[INFO] Downloading SAM checkpoint from:"
  echo "       $MODEL_URL"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --progress-bar "$MODEL_URL" -o "$MODEL_FILE"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$MODEL_FILE" "$MODEL_URL"
  else
    echo "[FAIL] curl or wget is required to download model files." >&2
    exit 1
  fi
fi

if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL_SHA256="$(sha256sum "$MODEL_FILE" | awk '{print $1}')"
  if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
    echo "[FAIL] SAM checkpoint checksum mismatch." >&2
    echo "       expected: $EXPECTED_SHA256" >&2
    echo "       actual:   $ACTUAL_SHA256" >&2
    exit 1
  fi
  echo "[OK] Checksum verified: $MODEL_FILE"
else
  echo "[WARN] sha256sum not available; skipped checksum verification." >&2
fi
