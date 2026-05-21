import argparse
import concurrent.futures
import json
import os
import re
from datetime import datetime

import requests
from tqdm import tqdm
from transformers import AutoTokenizer


PROMPT_DIRECT = """I will give you several history chats between you and a user. Please answer the question based on the relevant chat history.


History Chats:

$DOC$

Current Date: $Q_DATE$
Question: $Q$
Answer:"""


GEMMA4_THOUGHT_RE = re.compile(
    r"<\|channel>thought\s*.*?<channel\|>\s*",
    flags=re.DOTALL,
)

GENERIC_THINK_RE = re.compile(
    r"<think>.*?</think>\s*",
    flags=re.DOTALL,
)

def extract_hypothesis(response: str) -> str:
    response = response.strip()

    # Gemma 4 thinking format:
    # <|channel>thought ... <channel|> final answer
    if "<|channel>thought" in response and "<channel|>" in response:
        return GEMMA4_THOUGHT_RE.sub("", response).strip()

    # Fallback for DeepSeek/Qwen-style models
    if "</think>" in response:
        return GENERIC_THINK_RE.sub("", response).strip()

    return response

def build_history_text(item):
    sessions = list(zip(item["haystack_dates"], item["haystack_sessions"]))

    def _date_key(x):
        raw = x[0]
        try:
            return (0, datetime.strptime(raw, "%Y/%m/%d (%a) %H:%M"))
        except Exception:
            return (1, str(raw))

    # longmemeval_oracle; sort by timestamp
    sessions.sort(key=_date_key)

    history_string = ""
    for i, (sess_date, sess_turns) in enumerate(sessions):
        sess_string = ""
        for turn in sess_turns:
            role = str(turn.get("role", "")).strip()
            content = str(turn.get("content", "")).strip()
            sess_string += f"\n\n{role}: {content}"

        history_string += (
            f"\n### Session {i+1}:\n"
            f"Session Date: {sess_date}\n"
            f"Session Content:\n"
            f"{sess_string}\n"
        )

    return history_string


def truncate_prompt(prompt, tokenizer, max_len):
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) <= max_len:
        return prompt
    token_ids = token_ids[: max_len // 2] + token_ids[-max_len // 2 :]
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_output_file(args):
    os.makedirs(args.save_dir, exist_ok=True)
    dataset_name = os.path.splitext(os.path.basename(args.data_file))[0]
    model_name = args.model_name.split("/")[-1]
    return os.path.join(args.save_dir, f"{model_name}_{dataset_name}_seed{args.seed}.jsonl")



def write_outputs(args, data, outputs):
    out_file = get_output_file(args)

    with open(out_file, "a", encoding="utf-8") as fout:
        for item, output in zip(data, outputs):
            response = output["text"].strip()
            if not response:
                continue

            hypothesis = extract_hypothesis(response)

            fout.write(
                json.dumps(
                    {
                        "question_id": item["question_id"],
                        "hypothesis": hypothesis,
                        "response": response,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fout.flush()


def build_request_payload(args, prompt):
    return {
        "model": args.model_name,
        "prompt": prompt,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_new_tokens,
        "skip_special_tokens": False,
        "seed": args.seed,
        "stream": False,
    }


def call_completion(args, url, prompt):
    response = requests.post(
        url,
        json=build_request_payload(args, prompt),
        timeout=6000,
    )
    response.raise_for_status()
    response_json = response.json()
    return {"text": response_json["choices"][0]["text"]}


def run_disaggregated_generation(args, data, prompts):
    url = args.server_url.rstrip("/") + "/v1/completions"
    outputs = [None] * len(prompts)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_concurrent_requests
    ) as executor:
        futures = {
            executor.submit(call_completion, args, url, prompt): idx
            for idx, prompt in enumerate(prompts)
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Inference",
        ):
            outputs[futures[future]] = future.result()
    write_outputs(args, data, outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/longmemeval_s_cleaned.json",
        help="Path to longmemeval_oracle.json or longmemeval_s_cleaned.json",
    )
    parser.add_argument("--save_dir", "-s", type=str, default="results")

    parser.add_argument("--max_context_len", type=int, default=120000)
    parser.add_argument("--max_new_tokens", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)

    # sample parameters
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--enable_thinking", action='store_true', default=False)

    parser.add_argument("--model_name", type=str, required=True, help="Model path/name")
    parser.add_argument("--server_url", type=str, default="http://localhost:8192")
    parser.add_argument("--max_concurrent_requests", type=int, default=8)
    
    args = parser.parse_args()

    print(args)

    data = load_data(args.data_file)

    out_file = get_output_file(args)

    # cache
    existing_qids = set()
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as fin:
            existing_qids = {
                json.loads(line)["question_id"]
                for line in fin
                if line.strip()
            }
    data = [item for item in data if item["question_id"] not in existing_qids]
    if not data:
        print("All selected data already exists, skip generation.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    prompts = []
    for item in data:
        history = build_history_text(item)
        prompt = PROMPT_DIRECT.replace("$DOC$", history).replace("$Q_DATE$", item["question_date"]).replace("$Q$", item["question"])
        prompt = truncate_prompt(prompt, tokenizer, args.max_context_len)

        messages = [{"role": "user", "content": prompt}]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=args.enable_thinking))

    run_disaggregated_generation(args, data, prompts)


if __name__ == "__main__":
    main()
