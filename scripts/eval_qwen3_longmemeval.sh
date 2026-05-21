#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_DIR="$REPO_ROOT/evaluation/LongMemEval"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
SERVER_URL="${SERVER_URL:-http://localhost:8595}"
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-32}"
DATA_FILE="${DATA_FILE:-data/longmemeval_s_cleaned.json}"
SAVE_DIR="${SAVE_DIR:-results/qwen3-8b}"
SEEDS_STR="${SEEDS:-42}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-112000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-19072}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"
JUDGE_MODEL="${JUDGE_MODEL:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval_qwen3_longmemeval.sh [options]

Options:
  --model-name <name>              model name used by tokenizer and server request.
  --server-url <url>               OpenAI-compatible server URL.
  --max-concurrent-requests <n>    request concurrency.
  --data-file <path>               LongMemEval data file.
  --save-dir <dir>                 output directory under evaluation/LongMemEval.
  --seed <n>                       run one seed.
  --seeds "<n n ...>"              run multiple seeds.
  --max-context-len <n>            max context length.
  --max-new-tokens <n>             max generated tokens.
  --temperature <v>                sampling temperature.
  --top-p <v>                      sampling top_p.
  --top-k <n>                      sampling top_k.
  --enable-thinking                enable Qwen thinking template flag.
  --disable-thinking               disable Qwen thinking template flag.
  --judge-model <name>             optional LongMemEval judge model, e.g. gpt-4o.
  -h, --help                       show this help.
EOF
}

while (($# > 0)); do
  case "$1" in
    --model-name|--model) MODEL_NAME="${2:?missing value for $1}"; shift 2 ;;
    --server-url) SERVER_URL="${2:?missing value for --server-url}"; shift 2 ;;
    --max-concurrent-requests) MAX_CONCURRENT_REQUESTS="${2:?missing value for --max-concurrent-requests}"; shift 2 ;;
    --data-file) DATA_FILE="${2:?missing value for --data-file}"; shift 2 ;;
    --save-dir) SAVE_DIR="${2:?missing value for --save-dir}"; shift 2 ;;
    --seed) SEEDS_STR="${2:?missing value for --seed}"; shift 2 ;;
    --seeds) SEEDS_STR="${2:?missing value for --seeds}"; shift 2 ;;
    --max-context-len) MAX_CONTEXT_LEN="${2:?missing value for --max-context-len}"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="${2:?missing value for --max-new-tokens}"; shift 2 ;;
    --temperature) TEMPERATURE="${2:?missing value for --temperature}"; shift 2 ;;
    --top-p) TOP_P="${2:?missing value for --top-p}"; shift 2 ;;
    --top-k) TOP_K="${2:?missing value for --top-k}"; shift 2 ;;
    --enable-thinking) ENABLE_THINKING=1; shift ;;
    --disable-thinking) ENABLE_THINKING=0; shift ;;
    --judge-model) JUDGE_MODEL="${2:?missing value for --judge-model}"; shift 2 ;;
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

if [[ ! -f "$DATA_FILE" ]]; then
  echo "LongMemEval data file not found: ${EVAL_DIR}/${DATA_FILE}" >&2
  echo "Set DATA_FILE or pass --data-file after preparing the LongMemEval JSON file." >&2
  exit 1
fi

for seed in "${SEEDS[@]}"; do
  echo "Running LongMemEval: data=${DATA_FILE}, seed=${seed}"
  args=(
    --model_name "$MODEL_NAME"
    --server_url "$SERVER_URL"
    --max_concurrent_requests "$MAX_CONCURRENT_REQUESTS"
    --data_file "$DATA_FILE"
    --save_dir "$SAVE_DIR"
    --seed "$seed"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --top_k "$TOP_K"
    --max_context_len "$MAX_CONTEXT_LEN"
    --max_new_tokens "$MAX_NEW_TOKENS"
  )
  if [[ "$ENABLE_THINKING" == "1" ]]; then
    args+=(--enable_thinking)
  fi

  python3 pred_mix.py "${args[@]}"

  if [[ -n "$JUDGE_MODEL" ]]; then
    dataset_name="$(basename "$DATA_FILE")"
    dataset_name="${dataset_name%.*}"
    model_short_name="${MODEL_NAME##*/}"
    hyp_file="${SAVE_DIR}/${model_short_name}_${dataset_name}_seed${seed}.jsonl"
    python3 src/evaluation/evaluate_qa.py "$JUDGE_MODEL" "$hyp_file" "$DATA_FILE"
  fi
done
