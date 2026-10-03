# Lmaana Assistant

Darija voice RAG assistant for Moroccan public-service information.

Lmaana Assistant turns spoken Moroccan Darija into a grounded answer:

```text
microphone → ASR → Darija normalization → intent detection
           → document retrieval → LLM → Darija answer
```

## Project goals

- Compare Darija-capable ASR models, including Lmaana and Whisper.
- Build a searchable knowledge base from public Moroccan administrative and service documentation.
- Generate answers grounded in retrieved sources rather than unsupported model memory.
- Measure ASR quality, retrieval quality, faithfulness, and end-to-end latency.

## Planned stack

- Python and FastAPI for the backend
- LangGraph/LangChain for orchestration
- Qwen3-Embedding with FAISS or Qdrant for retrieval
- Quantized Qwen3-4B for local answer generation
- Streamlit or React for the user interface

## Repository layout

```text
src/sawtma/
  api/          HTTP endpoints
  asr/          speech-to-text adapters
  retrieval/    ingestion, chunking, and vector search
  generation/   grounded answer generation
  evaluation/   WER/CER, retrieval, faithfulness, and latency metrics
  config.py     application configuration
tests/          automated tests
data/           local datasets and indexes (ignored by Git)
```

## Quick start

Requires Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

The application code is intentionally minimal at this stage. Components will be added in small, testable increments.

## Status

Repository foundation created. The next milestone is a minimal vertical slice: upload or record audio, transcribe it, retrieve one relevant document, and return a cited answer.
