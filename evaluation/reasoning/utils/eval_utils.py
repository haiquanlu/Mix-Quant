# utils/eval_utils.py
# -*- coding: utf-8 -*-
import os
import re
import sys
from typing import Optional, Iterator, List, Tuple

from .math_eval.grader import grade_answer
from datasets import load_dataset  # 放这里，避免 import 冲突

def gold_from_gsm8k(ex: dict) -> str:
    ans = ex.get("answer", "")
    parts = ans.split("####")
    return parts[-1].strip() if parts else ans.strip()

def gold_from_math500(ex: dict) -> str:
    return str(ex.get("answer", "")).strip()

def gold_from_aime(ex: dict) -> str:
    return str(ex.get("answer", "")).strip()

def gold_from_gpqa(_: dict) -> str:
    # 你在 prompt 中把正确选项固定放在 A
    return "A"

def load_gold_dataset(dataset: str):
    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main")["test"]
        key_q = "question"
        gold_fn = gold_from_gsm8k
    elif dataset == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", "default")["test"]
        key_q = "problem"
        gold_fn = gold_from_math500
    elif dataset == "aime24":
        ds = load_dataset("HuggingFaceH4/aime_2024", "default")["train"]
        key_q = "problem"
        gold_fn = gold_from_aime
    elif dataset == "aime25":
        ds = load_dataset("math-ai/aime25", "default")["test"]
        key_q = "problem"
        gold_fn = gold_from_aime
    elif dataset == "gpqa":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
        key_q = "Question"
        gold_fn = gold_from_gpqa
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return ds, key_q, gold_fn

def build_gold_maps(ds, key_q: str, gold_fn):
    mp = {}
    idx_map = {}
    for i, ex in enumerate(ds):
        q = str(ex.get(key_q, "")).strip()
        mp[q] = gold_fn(ex)
        idx_map[q] = i
    return mp, idx_map


# ----------------------------
# Balanced-brace macro extractor
# ----------------------------
def iter_macro_contents(s: str, macro: str) -> Iterator[str]:
    r"""
    Yield contents of all occurrences of \macro{...} with nested-brace matching.
    Example:
        iter_macro_contents(r"\boxed{(3, \frac{\pi}{2})} \boxed{1}", "boxed")
        -> "(3, \frac{\pi}{2})", "1"
    """
    if not s:
        return
    pat = f"\\{macro}{{"
    i = 0
    n = len(s)
    while True:
        i = s.find(pat, i)
        if i == -1:
            break
        j = i + len(pat)
        depth = 1
        k = j
        while k < n and depth > 0:
            ch = s[k]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            k += 1
        if depth == 0:
            inner = s[j:k-1]
            yield inner
            i = k
        else:
            break

# GPQA letter patterns
_GPQA_PATTERNS = [
    re.compile(r"(?i)\\boxed\{\(?([A-D])\)?\}"),
    re.compile(r"(?i)final\s*answer\s*[:：]?\s*([A-D])"),
    re.compile(r"(?i)answer\s*is\s*[:：]?\s*([A-D])"),
    re.compile(r"(?i)\banswer\b\s*[:：]?\s*([A-D])"),
]

_MATHY_SPAN = re.compile(
    r"(\\left\s*\(.*?\\right\s*\)|\([^\)]*?\)|\\frac\{.*?\}\{.*?\}|\\sqrt\{.*?\}|[-+]?\d+\s*/\s*[-+]?\d+|[-+]?\d+(?:\.\d+)?)",
    flags=re.DOTALL,
)

def _unwrap_text_macro(s: str) -> str:
    s = s.strip()
    if s.startswith(r"\text{"):
        inner = next(iter_macro_contents(s, "text"), None)
        if inner is not None and len(inner) + len(r"\text{}") + 0 >= len(s) - 2:
            return inner.strip()
    return s

def extract_math_pred(text: str) -> Optional[str]:
    if not text:
        return None
    boxed = list(iter_macro_contents(text, "boxed"))
    if boxed:
        candidate = boxed[-1].strip()
        candidate = _unwrap_text_macro(candidate)
        return candidate if candidate else None

    for pat in [
        re.compile(r"(?i)final\s*answer\s*[:：]?\s*([^\n\r\.;]+)"),
        re.compile(r"(?i)answer\s*is\s*[:：]?\s*([^\n\r\.;]+)"),
        re.compile(r"(?i)\banswer\b\s*[:：]?\s*([^\n\r\.;]+)"),
    ]:
        m = pat.search(text)
        if m:
            cand = m.group(1).strip()
            return cand if cand else None

    last = None
    for m in _MATHY_SPAN.finditer(text):
        last = m.group(1)
    if last:
        return last.strip()
    return None

def extract_gpqa_pred(text: str) -> Optional[str]:
    if not text:
        return None
    for pat in _GPQA_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    tail = text.strip()[-20:]
    m = re.search(r"([A-D])(?!.*[A-D])", tail) or re.search(r"([A-D])(?!.*[A-D])", text)
    return m.group(1).upper() if m else None


# ----------------------------
# In-memory evaluation (no CSV)
# ----------------------------
def evaluate_predictions(
    dataset: str,
    questions: List[str],
    answers: List[str],
    lengths: Optional[List[Optional[int]]] = None,
) -> dict:
    """
    直接用内存里的 (questions, answers, lengths) 计算准确率，并可保存 acc_and_length.txt / wrong_indices.txt
    返回 metrics 字典。
    """
    assert len(questions) == len(answers), "questions 与 answers 数量不一致"
    total = len(questions)

    ds, key_q, gold_fn = load_gold_dataset(dataset)
    gold_map, gold_idx_map = build_gold_maps(ds, key_q, gold_fn)

    preds: List[str] = []
    golds: List[str] = []
    oks:   List[bool] = []
    dataset_indices: List[Optional[int]] = []
    extracted = 0

    for q, raw in zip(questions, answers):
        q = (q or "").strip()
        raw = "" if raw is None else str(raw)
        gold = gold_map.get(q, None)
        dataset_indices.append(gold_idx_map.get(q, None))

        if dataset == "gpqa":
            pred = extract_gpqa_pred(raw)
            ok = (pred is not None) and (pred.upper() == "A")
        else:
            pred = extract_math_pred(raw)
            if pred is None or gold is None:
                ok = False
            else:
                ok = bool(grade_answer(pred, gold))

        if pred is not None:
            extracted += 1

        preds.append("" if pred is None else pred)
        golds.append("" if gold is None else gold)
        oks.append(ok)

    acc = float(sum(oks)) / total if total else 0.0
    ext_rate = float(extracted) / total if total else 0.0

    # 平均长度
    avg_len = None
    if lengths is not None and len(lengths) == total:
        nums = [x for x in lengths if isinstance(x, (int, float))]
        if nums:
            avg_len = float(sum(nums) / len(nums))

    return {
        "acc": acc,
        "extracted_rate": ext_rate,
        "avg_length": avg_len,
    }