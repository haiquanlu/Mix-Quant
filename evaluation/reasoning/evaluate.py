import random
import torch
import numpy as np

import os
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import pandas as pd
from datasets import load_dataset

from utils.eval_utils import evaluate_predictions


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_chat(args, tokenizer, user_prompt):

    instruction = "Put your final answer within \\boxed{}."
    messages = [
        {"role": "user", "content": f"{instruction}\n{user_prompt}"},
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=args.enable_thinking
    )


if __name__ == "__main__":
    # read seed from argument
    args = argparse.ArgumentParser()
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--dataset", type=str, required=True)
    args.add_argument("--model_name", type=str, required=True)

    args.add_argument("--tensor_parallel_size", type=int, default=1)

    args.add_argument("--temperature", type=float, default=0.6)
    args.add_argument("--top_p", type=float, default=0.95)
    args.add_argument("--top_k", type=int, default=20)
    args.add_argument("--max_tokens", type=int, default=32768)
    args.add_argument("--max_model_length", type=int, default=49152)

    args.add_argument("--enable_thinking", action='store_true', default=False)



    args = args.parse_args()

    set_seed(args.seed)


    # dataset loading
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

    # model        
    model_name = args.model_name.split("/")[-1]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # use vLLM
    sampling_params = SamplingParams(
        temperature=args.temperature, top_p = args.top_p, top_k=args.top_k, 
        max_tokens=args.max_tokens, 
        seed=args.seed,
    )
    
    model = LLM(args.model_name, trust_remote_code=True, tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_length, gpu_memory_utilization=0.95)



    # build all prompts
    raw_questions = [i[question_key] for i in dataset]
    prompts = [build_chat(args, tokenizer, q) for q in raw_questions]

    # vllm continuous batching
    outputs = model.generate(
        prompts,
        sampling_params,
    )

    # save results
    result_dir = os.path.join('results/baselines', model_name)
    if args.enable_thinking:
        result_dir = os.path.join(result_dir, 'thinking')
    else:
        result_dir = os.path.join(result_dir, 'no_thinking')

    os.makedirs(result_dir, exist_ok=True)
    result_file_name = os.path.join(result_dir, f"{args.dataset}.csv")

    results = []
    answers = []    
    lengths = []
    for q, out in zip(raw_questions, outputs):
        text = out.outputs[0].text
        tok_ids = out.outputs[0].token_ids
        length = len(tok_ids) if tok_ids is not None else None
        results.append({"Question": q, "Answer": text, "length": length})
        answers.append(text)
        lengths.append(length)

    # save responses
    # result_df = pd.DataFrame(results)
    # result_df.to_csv(result_file_name, index=False)
     
    #  evaluate metrics
    output_path = os.path.join(result_dir, f"{args.dataset}_seed{args.seed}_metrics.txt")
    metrics = evaluate_predictions(
        dataset=args.dataset,
        questions=raw_questions,
        answers=answers,
        lengths=lengths,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for key in ("acc", "extracted_rate", "avg_length"):
            f.write(f"{key}: {metrics.get(key)}\n")