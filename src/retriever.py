"""
Baseline retriever: embeds each context document once per question and
retrieves the top-k most similar documents to the raw question text.

This is intentionally naive single-shot retrieval — no query reformulation,
no multi-hop logic. It exists to give the baseline pipeline (and later, the
agent's *first* retrieval call) a consistent, simple retrieval mechanism.
"""
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

_model = None

def get_embedder():
    """Lazy-load the embedding model once per process (it's ~80MB)."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def build_index(context: dict):
    """
    Builds a FAISS flat index over one HotpotQA example's context documents.

    `context` is HotpotQA's raw format: {"title": [...], "sentences": [[...], [...]]}
    Each document's sentences are joined into one string and embedded as a
    single vector -- document-level retrieval, not sentence-level.

    Returns: (index, titles, doc_texts) so callers can map retrieved
    positions back to readable document titles/content.
    """
    model = get_embedder()
    titles = context["title"]
    doc_texts = [" ".join(sentences) for sentences in context["sentences"]]

    embeddings = model.encode(doc_texts, convert_to_numpy=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))

    return index, titles, doc_texts


def retrieve(query: str, index, titles, doc_texts, top_k: int = 2):
    """
    Retrieves the top_k documents most similar to `query`.

    Returns a list of (title, text) tuples, ranked most-similar first.
    """
    model = get_embedder()
    query_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)

    distances, indices = index.search(query_vec, top_k)
    results = [(titles[i], doc_texts[i]) for i in indices[0]]
    return results