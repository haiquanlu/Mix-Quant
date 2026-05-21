import argparse
import concurrent.futures
import os

import requests
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.eval_utils import evaluate_predictions


def build_chat(args, tokenizer, user_prompt):
    instruction = "Put your final answer within \\boxed{}."
    messages = [
        {"role": "user", "content": f"{instruction}\n{user_prompt}"},
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=args.enable_thinking
    )


def build_request_payload(args, prompt):
    return {
        "model": args.model_name,
        "prompt": prompt,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "stream": False,
    }


def save_decode_results(args, raw_questions, outputs):
    result_model_name = args.output_model_name or args.model_name
    model_name = result_model_name.split("/")[-1]
    result_dir = os.path.join('results', model_name)
    if args.enable_thinking:
        result_dir = os.path.join(result_dir, 'thinking')
    else:
        result_dir = os.path.join(result_dir, 'no_thinking')

    os.makedirs(result_dir, exist_ok=True)

    answers = []
    lengths = []
    for out in outputs:
        text = out["text"]
        length = out["length"]
        answers.append(text)
        lengths.append(length)

    # result_file_name = os.path.join(result_dir, f"{args.dataset}_seed{args.seed}.csv")
    # results = [{"Question": q, "Answer": a, "length": l}
    #            for q, a, l in zip(raw_questions, answers, lengths)]
    # result_df = pd.DataFrame(results)
    # result_df.to_csv(result_file_name, index=False)

    output_path = os.path.join(result_dir, f"{args.dataset}_seed{args.seed}_metrics.txt")
    metrics = evaluate_predictions(
        dataset=args.dataset,
        questions=raw_questions,
        answers=answers,
        lengths=lengths,
    )

    with open(output_path, "a", encoding="utf-8") as f:
        for key in ("acc", "extracted_rate", "avg_length"):
            f.write(f"{key}: {metrics.get(key)}\n")
        f.write("\n")


def call_completion(args, url, prompt):
    response = requests.post(
        url,
        json=build_request_payload(args, prompt),
        timeout=6000,
    )
    response.raise_for_status()
    response_json = response.json()
    text = response_json["choices"][0]["text"]
    usage = response_json.get("usage") or {}
    return {
        "text": text,
        "length": usage.get("completion_tokens"),
    }


def run_disaggregated_generation(args, raw_questions, prompts):
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
    save_decode_results(args, raw_questions, outputs)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--dataset", type=str, required=True)

    # sampling parameters
    args.add_argument("--temperature", type=float, default=0.6)
    args.add_argument("--top_p", type=float, default=0.95)
    args.add_argument("--top_k", type=int, default=20)
    args.add_argument("--max_tokens", type=int, default=32768)
    args.add_argument("--enable_thinking", action='store_true', default=False)

    args.add_argument("--model_name", type=str, required=True)
    args.add_argument("--output_model_name", type=str, default=None)
    args.add_argument("--server_url", type=str, default="http://localhost:8192")
    args.add_argument("--max_concurrent_requests", type=int, default=8)

    args = args.parse_args()

    if args.dataset == 'gsm8k':
        dataset = load_dataset('openai/gsm8k', 'main')['test']
        question_key = 'question'
    elif args.dataset == 'aime24':
        dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]
        question_key = 'problem'
    elif args.dataset == 'aime25':
        dataset = load_dataset("math-ai/aime25")["test"]
        question_key = 'problem'
    elif args.dataset == 'math500':
        dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
        question_key = 'problem'
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    raw_questions = [i[question_key] for i in dataset]
    prompts = [build_chat(args, tokenizer, q) for q in raw_questions]

    run_disaggregated_generation(args, raw_questions, prompts)
