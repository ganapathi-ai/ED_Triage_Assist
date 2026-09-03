# ED Triage Assist — AI-Powered RAG Platform

End-to-end RAG application for Emergency Department Triage with maximum retrieval techniques.

## Architecture

### RAG Techniques Implemented
1. **Hybrid Search** — BM25 (sparse) + Vector (dense) with Reciprocal Rank Fusion
2. **Query Expansion** — Domain-specific synonym expansion for ED/triage terminology
3. **HyDE** — Hypothetical Document Embeddings for vocabulary gap bridging
4. **Multi-Query Fusion** — Multiple query variations fused with RRF
5. **Cross-Encoder Reranking** — ms-marco-MiniLM cross-encoder for relevance scoring
6. **MMR** — Maximal Marginal Relevance for diverse, non-redundant results
7. **Parent-Child Chunking** — Large parent chunks for context, small child chunks for retrieval
8. **Conversational Memory** — Multi-turn context with session management
9. **Contextual Retrieval** — Follow-up question detection and context injection
10. **Citation Tracking** — Source attribution in every response
11. **RAG Evaluation** — Retrieval quality and answer faithfulness metrics

### Tech Stack
- **Backend**: FastAPI + Python
- **Vector DB**: ChromaDB
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Sparse Search**: BM25Okapi
- **Reranking**: Cross-Encoder (ms-marco-MiniLM-L-6-v2)
- **LLM**: OpenAI GPT-4o-mini / Anthropic Claude / Groq
- **Frontend**: Three.js 3D Dashboard + RAG Chat + ESI Assistant

### Project Structure
```
backend/
  app/
    api/main.py          — FastAPI endpoints
    config.py            — Configuration management
    services/
      document_processor.py  — PDF, PPTX, DOCX processing
      chunker.py             — Parent-child & semantic chunking
      embeddings.py          — Sentence transformer embeddings
      hybrid_search.py       — BM25 + Vector + RRF fusion
      reranker.py            — Cross-encoder + MMR
      query_enhancer.py      — Query expansion, HyDE, multi-query
      llm_service.py         — Multi-provider LLM interface
      vector_store.py        — ChromaDB management
      rag_pipeline.py        — Main RAG orchestration
      memory.py              — Conversation memory
      evaluator.py           — RAG evaluation metrics
  scripts/ingest.py       — Document ingestion pipeline
  requirements.txt        — Python dependencies
  .env                    — Environment config
webapp/
  public/index.html       — 3D Dashboard + RAG UI
data/                     — Source documents (PDFs, PPTX)
docs/                     — Generated documentation
```

## Setup

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure API Keys
Edit `backend/.env` with your API keys (Groq, OpenAI, Anthropic).

### 3. Ingest Documents
```bash
cd backend
python scripts/ingest.py
```

### 4. Start Backend
```bash
cd backend
python run.py
```

### 5. Open Frontend
Open `webapp/public/index.html` in a browser (or serve via any HTTP server).

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed system health |
| `/api/chat` | POST | RAG query with full pipeline |
| `/api/ingest` | POST | Process and ingest documents |
| `/api/stats` | GET | Pipeline metrics |
| `/api/sessions/{id}/history` | GET | Conversation history |
| `/api/sessions/{id}` | DELETE | Clear session |

## Keyboard Shortcuts
- `1-5`: Switch 3D views
- `C`: Open AI Chat sidebar
- `E`: Open ESI Triage Assistant
- `S`: Open Document Search
- `Esc`: Close panels

## RAG Pipeline Flow
```
User Query
    ↓
Query Enhancement (expansion + HyDE + multi-query)
    ↓
Hybrid Search (BM25 + Vector → RRF fusion)
    ↓
HyDE Retrieval (augment with hypothetical embeddings)
    ↓
Cross-Encoder Reranking
    ↓
MMR Diversity Selection
    ↓
Context Assembly + LLM Generation
    ↓
Answer with Citations
```
