"""Load and lightly preprocess a HotpotQA slice for the project."""
from datasets import load_dataset

def load_hotpotqa(split: str = "validation", n: int = 200, seed: int = 42):
    """
    Loads a subset of HotpotQA (distractor config) for fast local iteration.
    n=200 keeps eval runs cheap while still being statistically meaningful
    enough to compare baseline vs. agentic approaches.
    """
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    ds = ds.shuffle(seed=seed).select(range(n))
    return ds

if __name__ == "__main__":
    data = load_hotpotqa()
    print(f"Loaded {len(data)} examples")
    example = data[0]
    print("Question:", example["question"])
    print("Answer:", example["answer"])
    print("Num supporting facts:", len(example["supporting_facts"]["sent_id"]))
    print("Num context docs:", len(example["context"]["title"]))