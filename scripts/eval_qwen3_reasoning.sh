#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_DIR="$REPO_ROOT/evaluation/reasoning"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
OUTPUT_MODEL_NAME="${OUTPUT_MODEL_NAME:-}"
SERVER_URL="${SERVER_URL:-http://localhost:8595}"
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-64}"
DATASETS_STR="${DATASETS:-math500 aime24 aime25}"
SEEDS_STR="${SEEDS:-42}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval_qwen3_reasoning.sh [options]

Options:
  --model-name <name>              model name used by tokenizer and server request.
  --output-model-name <name>       optional result directory model name.
  --server-url <url>               OpenAI-compatible server URL.
  --max-concurrent-requests <n>    request concurrency.
  --dataset <name>                 run one dataset.
  --datasets "<name name ...>"     run multiple datasets.
  --seed <n>                       run one seed.
  --seeds "<n n ...>"              run multiple seeds.
  --max-tokens <n>                 max generated tokens.
  --temperature <v>                sampling temperature.
  --top-p <v>                      sampling top_p.
  --top-k <n>                      sampling top_k.
  --enable-thinking                enable Qwen thinking template flag.
  --disable-thinking               disable Qwen thinking template flag.
  -h, --help                       show this help.
EOF
}

while (($# > 0)); do
  case "$1" in
    --model-name|--model) MODEL_NAME="${2:?missing value for $1}"; shift 2 ;;
    --output-model-name) OUTPUT_MODEL_NAME="${2:?missing value for --output-model-name}"; shift 2 ;;
    --server-url) SERVER_URL="${2:?missing value for --server-url}"; shift 2 ;;
    --max-concurrent-requests) MAX_CONCURRENT_REQUESTS="${2:?missing value for --max-concurrent-requests}"; shift 2 ;;
    --dataset) DATASETS_STR="${2:?missing value for --dataset}"; shift 2 ;;
    --datasets) DATASETS_STR="${2:?missing value for --datasets}"; shift 2 ;;
    --seed) SEEDS_STR="${2:?missing value for --seed}"; shift 2 ;;
    --seeds) SEEDS_STR="${2:?missing value for --seeds}"; shift 2 ;;
    --max-tokens) MAX_TOKENS="${2:?missing value for --max-tokens}"; shift 2 ;;
    --temperature) TEMPERATURE="${2:?missing value for --temperature}"; shift 2 ;;
    --top-p) TOP_P="${2:?missing value for --top-p}"; shift 2 ;;
    --top-k) TOP_K="${2:?missing value for --top-k}"; shift 2 ;;
    --enable-thinking) ENABLE_THINKING=1; shift ;;
    --disable-thinking) ENABLE_THINKING=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

read -r -a DATASETS <<<"$DATASETS_STR"
read -r -a SEEDS <<<"$SEEDS_STR"
if ((${#DATASETS[@]} == 0)) || ((${#SEEDS[@]} == 0)); then
  echo "Datasets and seeds must be non-empty" >&2
  exit 2
fi

cd "$EVAL_DIR"

for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "Running reasoning: dataset=${dataset}, seed=${seed}"
    args=(
      --dataset "$dataset"
      --model_name "$MODEL_NAME"
      --max_tokens "$MAX_TOKENS"
      --temperature "$TEMPERATURE"
      --top_p "$TOP_P"
      --top_k "$TOP_K"
      --seed "$seed"
      --server_url "$SERVER_URL"
      --max_concurrent_requests "$MAX_CONCURRENT_REQUESTS"
    )
    if [[ -n "$OUTPUT_MODEL_NAME" ]]; then
      args+=(--output_model_name "$OUTPUT_MODEL_NAME")
    fi
    if [[ "$ENABLE_THINKING" == "1" ]]; then
      args+=(--enable_thinking)
    fi

    python3 evaluate_mix.py "${args[@]}"
  done
done
