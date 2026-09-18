"""
Runs the baseline pipeline across the full 200-example HotpotQA slice,
scoring each result with EM/F1 and saving incrementally so a crash or
rate-limit hit mid-run doesn't lose completed work.
"""
import time
import json
from pathlib import Path
from tqdm import tqdm
from src.baseline import answer_baseline
from src.evaluate import exact_match, f1_score

# Anchor to the repo root (parent of src/), regardless of caller's cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_baseline_eval(dataset, output_path="results/baseline_results.jsonl", delay_seconds=5):
    full_path = PROJECT_ROOT / output_path
    full_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: count how many results already exist, skip that many.
    already_done = 0
    if full_path.exists():
        with open(full_path, "r", encoding="utf-8") as f:
            already_done = sum(1 for _ in f)

    if already_done > 0:
        print(f"Resuming: {already_done} examples already completed, skipping them.")

    results = []
    # Load prior results back into memory so aggregate_results still gets everything
    if already_done > 0:
        with open(full_path, "r", encoding="utf-8") as f:
            results = [json.loads(line) for line in f]

    with open(full_path, "a", encoding="utf-8") as f:  # "a" = append, not overwrite
        remaining = dataset.select(range(already_done, len(dataset)))
        for example in tqdm(remaining, desc="Baseline eval"):
            result = answer_baseline(example)
            result["em"] = exact_match(result["generated_answer"], result["gold_answer"])
            result["f1"] = f1_score(result["generated_answer"], result["gold_answer"])

            f.write(json.dumps(result) + "\n")
            f.flush()
            results.append(result)

            time.sleep(delay_seconds)

    return results