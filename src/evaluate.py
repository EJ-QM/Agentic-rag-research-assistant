import re
import string

def normalize_answer(text: str) -> str:
    """
    Lowercases, strips punctuation, and collapses whitespace so that
    'Prussian.', 'prussian', and 'PRUSSIAN ' all compare as equal.
    This is standard practice for QA benchmarks -- surface-level
    differences shouldn't be scored as wrong answers.
    """
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def exact_match(prediction: str, gold: str) -> int:
    """Returns 1 if normalized strings match exactly, else 0."""
    return int(normalize_answer(prediction) == normalize_answer(gold))

from collections import Counter

def f1_score(prediction: str, gold: str) -> float:
    """
    Token-level F1 between prediction and gold answer. Handles multi-word
    answers more fairly than EM -- partial credit for partial overlap.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    # Edge case: both empty (rare, but avoids a misleading score)
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    # Edge case: one side empty, the other isn't -- no overlap possible
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)

import time
import json
from tqdm import tqdm
from src.baseline import answer_baseline
from src.evaluate import exact_match, f1_score

def aggregate_results(results: list[dict]) -> dict:
    """
    Computes overall EM/F1 plus a breakdown by retrieval success:
    did the retriever actually surface all supporting-fact documents,
    or not? This tests whether retrieval failure predicts answer failure.
    """
    n = len(results)
    mean_em = sum(r["em"] for r in results) / n
    mean_f1 = sum(r["f1"] for r in results) / n

    retrieval_success, retrieval_failure = [], []
    for r in results:
        needed = set(r["supporting_titles"])
        retrieved = set(r["retrieved_titles"])
        if needed.issubset(retrieved):
            retrieval_success.append(r)
        else:
            retrieval_failure.append(r)

    def avg_f1(group):
        return sum(r["f1"] for r in group) / len(group) if group else None

    return {
        "n_examples": n,
        "mean_em": mean_em,
        "mean_f1": mean_f1,
        "n_retrieval_success": len(retrieval_success),
        "n_retrieval_failure": len(retrieval_failure),
        "f1_when_retrieval_succeeded": avg_f1(retrieval_success),
        "f1_when_retrieval_failed": avg_f1(retrieval_failure),
    }