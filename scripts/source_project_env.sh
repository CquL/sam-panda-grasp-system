#!/usr/bin/env bash

# Source this file from the repository root:
#   source scripts/source_project_env.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
else
  printf '[WARN] No .env file found at %s\n' "$ENV_FILE" >&2
  printf '[WARN] Copy .env.example to .env and set DASHSCOPE_API_KEY first.\n' >&2
fi

export DASHSCOPE_BASE_URL="${DASHSCOPE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export VLM_MODEL="${VLM_MODEL:-qwen-vl-max}"
export ROS_PYTHON_EXEC="${ROS_PYTHON_EXEC:-/usr/bin/python3}"

if [ -z "${DASHSCOPE_API_KEY:-}" ] || [ "${DASHSCOPE_API_KEY}" = "replace_me" ]; then
  printf '[WARN] DASHSCOPE_API_KEY is not configured. VLM nodes will skip model calls or return empty boxes.\n' >&2
else
  printf '[OK] DASHSCOPE_API_KEY loaded.\n'
fi

printf '[OK] VLM_MODEL=%s\n' "$VLM_MODEL"
printf '[OK] DASHSCOPE_BASE_URL=%s\n' "$DASHSCOPE_BASE_URL"
printf '[OK] ROS_PYTHON_EXEC=%s\n' "$ROS_PYTHON_EXEC"
if [ -n "${ANYGRASP_PYTHON:-}" ]; then
  printf '[OK] ANYGRASP_PYTHON=%s\n' "$ANYGRASP_PYTHON"
else
  printf '[WARN] ANYGRASP_PYTHON is not set. Launch files will fall back to /usr/bin/python3.\n' >&2
fi
