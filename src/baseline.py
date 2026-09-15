"""
Baseline: single-shot retrieve-then-generate RAG.

Given a HotpotQA example, retrieves top-k documents based on the raw
question (no reformulation, no re-retrieval), then asks Gemini to answer
using only those documents. This is the pipeline the agentic version
(src/agent.py, built later) is compared against.
"""
import os
from google import genai
from dotenv import load_dotenv

from src.retriever import build_index, retrieve

load_dotenv()
_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

MODEL_NAME = "gemini-3.6-flash"

BASELINE_PROMPT = """Answer the question using ONLY the context below. \
If the context doesn't contain enough information to answer, say "Cannot determine from context."
Give a short, direct answer -- no explanation, just the answer itself.

Context:
{context}

Question: {question}

Answer:"""


def answer_baseline(example: dict, top_k: int = 2) -> dict:
    """
    Runs the full baseline pipeline on one HotpotQA example.

    Returns a dict with the retrieved docs, the generated answer, and the
    gold answer, so results can be inspected and scored downstream.
    """
    index, titles, doc_texts = build_index(example["context"])
    retrieved = retrieve(example["question"], index, titles, doc_texts, top_k=top_k)

    context_str = "\n\n".join(f"[{title}] {text}" for title, text in retrieved)
    prompt = BASELINE_PROMPT.format(context=context_str, question=example["question"])

    response = _client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    return {
        "question": example["question"],
        "gold_answer": example["answer"],
        "retrieved_titles": [t for t, _ in retrieved],
        "supporting_titles": example["supporting_facts"]["title"],
        "generated_answer": response.text.strip(),
    }