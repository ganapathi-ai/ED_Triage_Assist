"""ED Triage Assist — RAG Backend API"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
import logging
import time

from app.config import settings
from app.models.schemas import (
    ChatRequest, ChatResponse, Source,
    ESIRequest, ESIResponse, DeteriorationRequest, DeteriorationResponse,
    WaitTimeRequest, WaitTimeResponse, TriageRequest, TriageResponse,
    SearchRequest, SearchResponse, SearchResult, IngestResponse, HealthResponse
)
from app.models.esi_predictor import ESIPredictor
from app.models.deterioration_predictor import DeteriorationPredictor
from app.models.wait_time_predictor import WaitTimePredictor
from app.services.rag_pipeline import RAGPipeline
from app.services.rag_pipeline import (
    HybridSearchEngine, Reranker, QueryEnhancer,
    MemoryManager, CitationTracker
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ED Triage Assist API",
    description="RAG-powered Emergency Department Triage Assistant with AI predictions",
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Lazy-loaded singletons ───────────────────────────────────────────────
_rag_pipeline: RAGPipeline | None = None
_esi_predictor: ESIPredictor | None = None
_deterioration_predictor: DeteriorationPredictor | None = None
_wait_time_predictor: WaitTimePredictor | None = None


def get_rag() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        logger.info("Initializing RAG pipeline...")
        _rag_pipeline = RAGPipeline()
        _rag_pipeline.initialize()
    return _rag_pipeline


def get_esi() -> ESIPredictor:
    global _esi_predictor
    if _esi_predictor is None:
        _esi_predictor = ESIPredictor()
    return _esi_predictor


def get_deterioration() -> DeteriorationPredictor:
    global _deterioration_predictor
    if _deterioration_predictor is None:
        _deterioration_predictor = DeteriorationPredictor()
    return _deterioration_predictor


def get_wait_time() -> WaitTimePredictor:
    global _wait_time_predictor
    if _wait_time_predictor is None:
        _wait_time_predictor = WaitTimePredictor()
    return _wait_time_predictor


# ─── Health ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", service="ED Triage Assist", version="5.0.0")


# ─── Chat ─────────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start = time.time()
    try:
        pipeline = get_rag()
        result = pipeline.query(
            question=request.question,
            conversation_history=None,
            filters={},
        )
        sources = [
            Source(
                document=s.get("source", ""),
                page=s.get("page"),
                excerpt=s.get("text", "")[:300],
                relevance=float(s.get("rerank_score", s.get("score", 0.0))),
            )
            for s in result.get("sources", [])[:5]
        ]
        return ChatResponse(
            answer=result.get("answer", ""),
            confidence=float(result.get("confidence", 0.0)),
            sources=sources,
            latency_ms=round((time.time() - start) * 1000, 1),
        )
    except Exception as exc:
        logger.error(f"Chat error: {exc}")
        return ChatResponse(
            answer=f"An error occurred: {str(exc)}. Please try again.",
            confidence=0.0,
            sources=[],
            latency_ms=round((time.time() - start) * 1000, 1),
        )


# ─── Predict: ESI ────────────────────────────────────────────────────────

@app.post("/api/predict/esi", response_model=ESIResponse)
async def predict_esi(request: ESIRequest):
    start = time.time()
    try:
        predictor = get_esi()
        result = predictor.predict(
            age=request.age,
            chief_complaint=request.chief_complaint,
            vital_signs={
                "bp_systolic": request.bp_systolic,
                "bp_diastolic": request.bp_diastolic,
                "hr": request.hr,
                "rr": request.rr,
                "temp": request.temp,
                "spo2": request.spo2,
            },
            medical_history=request.medical_history,
            presenting_symptoms=request.presenting_symptoms,
        )
        return ESIResponse(
            esi_level=result.get("esi_level", 3),
            confidence=float(result.get("confidence", 0.0)),
            reasoning=result.get("reasoning", ""),
            recommended_wait_time=result.get("recommended_wait_time", "30-60 minutes"),
            red_flags=result.get("red_flags", []),
            protocol=result.get("protocol", ""),
            latency_ms=round((time.time() - start) * 1000, 1),
        )
    except Exception as exc:
        logger.error(f"ESI prediction error: {exc}")
        return ESIResponse(
            esi_level=3, confidence=0.0,
            reasoning=f"Error: {str(exc)}",
            recommended_wait_time="Consult clinical staff",
            red_flags=[],
            protocol="Unable to determine",
            latency_ms=round((time.time() - start) * 1000, 1),
        )


# ─── Predict: Deterioration ──────────────────────────────────────────────

@app.post("/api/predict/deterioration", response_model=DeteriorationResponse)
async def predict_deterioration(request: DeteriorationRequest):
    start = time.time()
    try:
        predictor = get_deterioration()
        result = predictor.predict(
            age=request.age,
            vital_signs={
                "bp_systolic": request.bp_systolic,
                "bp_diastolic": request.bp_diastolic,
                "hr": request.hr,
                "rr": request.rr,
                "temp": request.temp,
                "spo2": request.spo2,
                "gcs": request.gcs,
            },
            medical_history=request.medical_history,
            presenting_symptoms=request.presenting_symptoms,
            current_medications=request.current_medications,
        )
        return DeteriorationResponse(
            risk_score=float(result.get("risk_score", 0.0)),
            risk_level=result.get("risk_level", "LOW"),
            qsofa_score=int(result.get("qsofa_score", 0)),
            deterioration_probability=float(result.get("deterioration_probability", 0.0)),
            warning_signs=result.get("warning_signs", []),
            monitoring_recommendations=result.get("monitoring_recommendations", []),
            time_window=result.get("time_window", "Stable - routine monitoring"),
            confidence=float(result.get("confidence", 0.0)),
            latency_ms=round((time.time() - start) * 1000, 1),
        )
    except Exception as exc:
        logger.error(f"Deterioration prediction error: {exc}")
        return DeteriorationResponse(
            risk_score=0.0, risk_level="UNKNOWN", qsofa_score=0,
            deterioration_probability=0.0, warning_signs=[],
            monitoring_recommendations=["Consult clinical staff immediately"],
            time_window="Unable to determine",
            confidence=0.0,
            latency_ms=round((time.time() - start) * 1000, 1),
        )


# ─── Predict: Wait Time ──────────────────────────────────────────────────

@app.post("/api/predict/wait-time", response_model=WaitTimeResponse)
async def predict_wait_time(request: WaitTimeRequest):
    start = time.time()
    try:
        predictor = get_wait_time()
        result = predictor.predict(
            esi_level=request.esi_level,
            hospital_load=request.hospital_load,
            department=request.department,
            day_of_week=request.day_of_week,
            hour_of_day=request.hour_of_day,
            current_queue_length=request.current_queue_length,
            available_doctors=request.available_doctors,
            patient_age=request.patient_age,
            chief_complaint=request.chief_complaint,
        )
        return WaitTimeResponse(
            predicted_wait_minutes=int(result.get("predicted_wait_minutes", 30)),
            confidence_interval_lower=int(result.get("ci_lower", 15)),
            confidence_interval_upper=int(result.get("ci_upper", 60)),
            factors=result.get("factors", []),
            recommendation=result.get("recommendation", "Please remain in waiting area"),
            confidence=float(result.get("confidence", 0.0)),
            latency_ms=round((time.time() - start) * 1000, 1),
        )
    except Exception as exc:
        logger.error(f"Wait time prediction error: {exc}")
        return WaitTimeResponse(
            predicted_wait_minutes=30,
            confidence_interval_lower=15,
            confidence_interval_upper=60,
            factors=["Unable to calculate factors"],
            recommendation="Please consult nursing staff for current wait times",
            confidence=0.0,
            latency_ms=round((time.time() - start) * 1000, 1),
        )


# ─── Predict: Full Triage ────────────────────────────────────────────────

@app.post("/api/predict/triage", response_model=TriageResponse)
async def predict_triage(request: TriageRequest):
    start = time.time()
    try:
        esi_result = await predict_esi(request)
        deter_result = await predict_deterioration(request)
        wait_result = await predict_wait_time(request)
        return TriageResponse(
            esi=esi_result,
            deterioration=deter_result,
            wait_time=wait_result,
            overall_priority=_compute_priority(esi_result, deter_result),
            latency_ms=round((time.time() - start) * 1000, 1),
        )
    except Exception as exc:
        logger.error(f"Full triage error: {exc}")
        raise


def _compute_priority(esi: ESIResponse, deter: DeteriorationResponse) -> str:
    if esi.esi_level == 1 or deter.risk_level in ("HIGH", "CRITICAL"):
        return "IMMEDIATE"
    if esi.esi_level == 2 or deter.risk_level == "MODERATE":
        return "URGENT"
    if esi.esi_level == 3:
        return "SEMI-URGENT"
    return "NON-URGENT"


# ─── Search ───────────────────────────────────────────────────────────────

@app.post("/api/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    start = time.time()
    try:
        pipeline = get_rag()
        results = pipeline.search(
            query=request.query,
            filters={},
            top_k=request.top_k or 5,
        )
        items = [
            SearchResult(
                document=r.get("source", ""),
                page=r.get("page"),
                excerpt=r.get("text", "")[:500],
                score=float(r.get("rerank_score", r.get("score", 0.0))),
            )
            for r in results[: request.top_k or 5]
        ]
        return SearchResponse(
            query=request.query,
            results=items,
            total_results=len(items),
            latency_ms=round((time.time() - start) * 1000, 1),
        )
    except Exception as exc:
        logger.error(f"Search error: {exc}")
        return SearchResponse(
            query=request.query, results=[], total_results=0,
            latency_ms=round((time.time() - start) * 1000, 1),
        )


# ─── Ingest ───────────────────────────────────────────────────────────────

@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_documents():
    start = time.time()
    try:
        pipeline = get_rag()
        result = pipeline.ingest_documents()
        return IngestResponse(
            documents_processed=int(result.get("documents_processed", 0)),
            chunks_created=int(result.get("chunks_created", 0)),
            status=result.get("status", "completed"),
            message=result.get("message", "Documents ingested successfully"),
            latency_ms=round((time.time() - start) * 1000, 1),
        )
    except Exception as exc:
        logger.error(f"Ingest error: {exc}")
        return IngestResponse(
            documents_processed=0, chunks_created=0,
            status="error",
            message=f"Ingestion failed: {str(exc)}",
            latency_ms=round((time.time() - start) * 1000, 1),
        )


# ─── Frontend ─────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    webapp_path = os.path.join(os.path.dirname(__file__), "..", "..", "webapp", "public", "index.html")
    if os.path.exists(webapp_path):
        return FileResponse(webapp_path)
    return JSONResponse({"message": "ED Triage Assist API running"}, status_code=200)


webapp_dir = os.path.join(os.path.dirname(__file__), "..", "..", "webapp", "public")
if os.path.isdir(webapp_dir):
    app.mount("/static", StaticFiles(directory=webapp_dir), name="static")
