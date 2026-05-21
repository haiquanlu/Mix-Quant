import argparse
import concurrent.futures
import json
import os
import re

import requests
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())

template_rag = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()


STRICT_PATTERNS = [
    re.compile(r"the\s+correct\s+answer\s+is\s*\(?\s*([A-D])\s*\)?", re.IGNORECASE),
    re.compile(r"final\s+answer\s*[:：]?\s*\(?\s*([A-D])\s*\)?", re.IGNORECASE),
    re.compile(r"\banswer\s*[:：]\s*\(?\s*([A-D])\s*\)?", re.IGNORECASE),
    re.compile(r"\boption\s*[:：]?\s*\(?\s*([A-D])\s*\)?", re.IGNORECASE),
]

GEMMA4_THOUGHT_BLOCK_RE = re.compile(
    r"<\|channel>thought\s*.*?<channel\|>\s*",
    flags=re.DOTALL,
)

def get_tail_text(text):
    if not isinstance(text, str):
        return text

    # 1. Gemma 4 thinking format:
    # <|channel>thought
    # ...
    # <channel|>
    # final answer
    matches = list(GEMMA4_THOUGHT_BLOCK_RE.finditer(text))
    if matches:
        return text[matches[-1].end():]

    # 2. Fallback
    gemma_end = "<channel|>"
    idx = text.rfind(gemma_end)
    if idx != -1:
        return text[idx + len(gemma_end):]

    # 3. DeepSeek/Qwen-style thinking format
    marker = "</think>"
    idx = text.rfind(marker)
    if idx != -1:
        return text[idx + len(marker):]
    return text

def extract_answer_cot(response):
    if not isinstance(response, str) or not response.strip():
        return None

    cleaned = response.replace('*', ' ')
    tail = get_tail_text(cleaned)
    candidates = []

    for pat in STRICT_PATTERNS:
        for match in pat.finditer(tail):
            candidates.append((match.start(), match.group(1).upper()))

    if not candidates:
        for pat in STRICT_PATTERNS:
            for match in pat.finditer(cleaned):
                candidates.append((match.start(), match.group(1).upper()))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None


def build_chat(tokenizer, user_prompt):
    messages = [
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=args.enable_thinking
    )

def prepare_prompt(prompt, model, tokenizer):
    max_len = maxlen_map[model]
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
        prompt = tokenizer.decode(input_ids, skip_special_tokens=False)
    return build_chat(tokenizer, prompt)


def write_outputs(data, contexts, outputs, args, fout):
    for item, context, output in zip(data, contexts, outputs):
        if not output or not output["text"]:
            continue

        response = output["text"].strip()
        item['response'] = response
        if args.cot:
            item['pred'] = extract_answer_cot(response)
        else:
            item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context[:1000]
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()


def build_request_payload(args, prompt):
    return {
        "model": args.model,
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


def run_disaggregated_generation(args, data, contexts, prompts, fout):
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
    write_outputs(data, contexts, outputs, args, fout)


def get_pred(data, args, fout):
    model = args.model
    if model not in model_map:
        raise ValueError(f"model must exist in config/model2path.json: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model_map[model], trust_remote_code=True)

    prompts = []
    contexts = []
    for item in data:
        context = item['context']

        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][:args.rag]
            retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
            context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot

        raw_prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        prompts.append(prepare_prompt(raw_prompt, model, tokenizer))
        contexts.append(context)

    run_disaggregated_generation(args, data, contexts, prompts, fout)

def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    seed_suffix = f"_seed{args.seed}"
    if args.rag > 0:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_rag_{str(args.rag)}{seed_suffix}.jsonl")
    elif args.no_context:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_no_context{seed_suffix}.jsonl")
    elif args.cot:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_cot{seed_suffix}.jsonl")
    else:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"{seed_suffix}.jsonl")

    dataset = load_dataset('THUDM/LongBench-v2', split='train') # dataset = json.load(open('data.json', 'r', encoding='utf-8'))
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]

    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    fout = open(out_file, 'a', encoding='utf-8')
    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)

    get_pred(data, args, fout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
    parser.add_argument("--no_context", "-nc", action='store_true') # set to True if using no context (directly measuring memorization)
    parser.add_argument("--rag", "-rag", type=int, default=0) # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--server_url", type=str, default="http://localhost:8192")
    parser.add_argument("--max_concurrent_requests", type=int, default=8)

    # sampling parameters
    parser.add_argument("--max_new_tokens", type=int, default=16384)
    parser.add_argument("--enable_thinking", action='store_true', default=False)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()
    main()