#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_DIR="$REPO_ROOT/evaluation/longbench-v2"

MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
SERVER_URL="${SERVER_URL:-http://localhost:8595}"
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-32}"
SAVE_DIR="${SAVE_DIR:-results/qwen3-8b}"
SEEDS_STR="${SEEDS:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-30720}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"
COT="${COT:-1}"
NO_CONTEXT="${NO_CONTEXT:-0}"
RAG="${RAG:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval_qwen3_longbench-v2.sh [options]

Options:
  --model <name>                   model key in evaluation/longbench-v2/config.
  --server-url <url>               OpenAI-compatible server URL.
  --max-concurrent-requests <n>    request concurrency.
  --save-dir <dir>                 output directory under evaluation/longbench-v2.
  --seed <n>                       run one seed.
  --seeds "<n n ...>"              run multiple seeds.
  --max-new-tokens <n>             max generated tokens.
  --temperature <v>                sampling temperature.
  --top-p <v>                      sampling top_p.
  --top-k <n>                      sampling top_k.
  --cot                            enable COT prompt.
  --no-cot                         disable COT prompt.
  --no-context                     enable no-context prompt.
  --rag <n>                        use top-n retrieved contexts.
  --enable-thinking                enable Qwen thinking template flag.
  --disable-thinking               disable Qwen thinking template flag.
  -h, --help                       show this help.
EOF
}

while (($# > 0)); do
  case "$1" in
    --model|--model-name) MODEL_NAME="${2:?missing value for $1}"; shift 2 ;;
    --server-url) SERVER_URL="${2:?missing value for --server-url}"; shift 2 ;;
    --max-concurrent-requests) MAX_CONCURRENT_REQUESTS="${2:?missing value for --max-concurrent-requests}"; shift 2 ;;
    --save-dir) SAVE_DIR="${2:?missing value for --save-dir}"; shift 2 ;;
    --seed) SEEDS_STR="${2:?missing value for --seed}"; shift 2 ;;
    --seeds) SEEDS_STR="${2:?missing value for --seeds}"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="${2:?missing value for --max-new-tokens}"; shift 2 ;;
    --temperature) TEMPERATURE="${2:?missing value for --temperature}"; shift 2 ;;
    --top-p) TOP_P="${2:?missing value for --top-p}"; shift 2 ;;
    --top-k) TOP_K="${2:?missing value for --top-k}"; shift 2 ;;
    --cot) COT=1; shift ;;
    --no-cot) COT=0; shift ;;
    --no-context) NO_CONTEXT=1; shift ;;
    --rag) RAG="${2:?missing value for --rag}"; shift 2 ;;
    --enable-thinking) ENABLE_THINKING=1; shift ;;
    --disable-thinking) ENABLE_THINKING=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

read -r -a SEEDS <<<"$SEEDS_STR"
if ((${#SEEDS[@]} == 0)); then
  echo "No seeds specified" >&2
  exit 2
fi

cd "$EVAL_DIR"

for seed in "${SEEDS[@]}"; do
  echo "Running LongBench-v2: model=${MODEL_NAME}, seed=${seed}"
  args=(
    --model "$MODEL_NAME"
    --server_url "$SERVER_URL"
    --max_concurrent_requests "$MAX_CONCURRENT_REQUESTS"
    --save_dir "$SAVE_DIR"
    --seed "$seed"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --top_k "$TOP_K"
  )
  if [[ "$COT" == "1" ]]; then
    args+=(--cot)
  fi
  if [[ "$NO_CONTEXT" == "1" ]]; then
    args+=(--no_context)
  fi
  if [[ "$RAG" != "0" ]]; then
    args+=(--rag "$RAG")
  fi
  if [[ "$ENABLE_THINKING" == "1" ]]; then
    args+=(--enable_thinking)
  fi

  python3 pred_mix.py "${args[@]}"
done
