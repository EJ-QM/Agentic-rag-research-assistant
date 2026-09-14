# Agentic Research Assistant (RAG + Multi-Tool Agent)

## Problem
Multi-hop question answering — where the answer requires combining evidence from
two or more documents — breaks naive single-shot RAG. This project builds and
evaluates an agent that can decide *when* to retrieve again, on the HotpotQA
benchmark.

## Approach
TBD — will document baseline vs. agentic approach after both are built and scored.

## Results
TBD

## Demo
TBD

## Limitations & Failure Cases
TBD

## Repo Structure
- `src/data.py` — HotpotQA loading and preprocessing
- `src/retriever.py` — vector retrieval over supporting docs
- `src/agent.py` — baseline + agentic tool-use implementations
- `src/evaluate.py` — EM/F1 scoring against HotpotQA gold answers
- `notebooks/research_log.ipynb` — full research log, in order
- `app/streamlit_demo.py` — interactive demo
