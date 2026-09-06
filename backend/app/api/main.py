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
from app.services.hybrid_search import HybridSearchEngine
from app.services.reranker import CrossEncoderReranker as Reranker
from app.services.query_enhancer import QueryEnhancer
from app.services.vector_store import vector_store
from app.services.memory import conversation_memory

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
        # Initialize vector store if needed
        if not vector_store.collection:
            vector_store.initialize()
        _rag_pipeline.initialize(vector_store.collection)
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
    try:
        vs = vector_store.get_stats()
    except Exception:
        vs = {"error": "unavailable"}
    try:
        mem = conversation_memory.get_stats()
    except Exception:
        mem = {"error": "unavailable"}
    return HealthResponse(
        status="ok",
        vector_store=vs,
        models={"embedding": "PubMedBERT", "llm": "gpt-4o-mini"},
        memory=mem,
        uptime_seconds=0.0,
    )


# ─── Chat ─────────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start = time.time()
    try:
        pipeline = get_rag()
        result = pipeline.query(
            user_query=request.query,
            session_id=request.session_id,
        )
        sources = [
            Source(
                source=s.get("source", ""),
                page=s.get("page"),
                excerpt=s.get("text_preview", "")[:300],
                relevance=float(s.get("score", 0.0)),
            )
            for s in result.sources[:5]
        ]
        return ChatResponse(
            answer=result.answer,
            confidence=float(result.confidence),
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
        from app.models.esi_predictor import ESIFeatures
        
        # Build features
        features = ESIFeatures(
            age=request.age,
            gender=getattr(request, 'gender', 'unknown'),
            hr=request.hr,
            bp_systolic=request.bp_systolic,
            bp_diastolic=request.bp_diastolic,
            spo2=request.spo2,
            rr=request.rr,
            temp=request.temp,
            glucose=None,
            gcs=getattr(request, 'gcs', 15),
            mental_status="alert",
            pain_score=None,
            chief_complaint=request.chief_complaint.lower(),
            mechanism_of_injury=None,
            has_allergy=len(request.medical_history) > 0,
            resource_count_estimate=2
        )
        result = predictor.predict(features)
        
        return ESIResponse(
            esi_level=result.get("level", 3),
            confidence=float(result.get("confidence", 0.0)),
            reasoning="; ".join(result.get("reasoning", [])),
            recommended_wait_time="30-60 minutes",
            red_flags=result.get("recommended_actions", []),
            protocol="Standard",
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
            patient_id="patient-1",
            vital_signs={
                "blood_pressure_systolic": request.bp_systolic,
                "blood_pressure_diastolic": request.bp_diastolic,
                "heart_rate": request.hr,
                "respiratory_rate": request.rr,
                "temperature": request.temp,
                "spo2": request.spo2,
                "gcs": getattr(request, "gcs", 15),
            },
            current_esi=getattr(request, "esi_level", 3),
            time_in_ed_minutes=15,
            chief_complaint=request.chief_complaint,
        )
        
        return DeteriorationResponse(
            risk_score=float(result.get("risk_score", 0.0)),
            risk_level=result.get("risk_level", "LOW"),
            qsofa_score=int(result.get("qsofa_score", 0)),
            deterioration_probability=float(result.get("risk_score", 0.0)),
            warning_signs=result.get("warning_signs", []),
            monitoring_recommendations=result.get("recommended_actions", []),
            time_window=f"Reassess in {result.get('time_to_reassess_minutes', 30)} min",
            confidence=0.85,
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
            patient_data={"esi_level": request.esi_level, "age": request.age, "arrival_mode": "walk-in"},
            ed_state={
                "current_ed_volume": max(10, request.current_queue_length * 4),
                "staff_on_duty": max(2, request.available_doctors + 2),
                "available_beds": max(0, 15 - request.current_queue_length)
            }
        )
        wait_mins = int(result.get("estimated_wait_minutes", 30))
        return WaitTimeResponse(
            predicted_wait_minutes=wait_mins,
            confidence_interval_lower=max(0, wait_mins - 15),
            confidence_interval_upper=wait_mins + 20,
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
        
        wait_req = WaitTimeRequest(
            esi_level=esi_result.esi_level,
            hospital_load=request.hospital_load,
            department=request.department,
            day_of_week=request.day_of_week,
            hour_of_day=request.hour_of_day,
            current_queue_length=request.current_queue_length,
            available_doctors=request.available_doctors,
            age=request.age,
            chief_complaint=request.chief_complaint
        )
        wait_result = await predict_wait_time(wait_req)
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
            top_k=request.top_k or 5,
        )
        items = [
            SearchResult(
                source=r.get("source", ""),
                page=r.get("page"),
                excerpt=r.get("text", "")[:500],
                score=float(r.get("score", 0.0)),
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
