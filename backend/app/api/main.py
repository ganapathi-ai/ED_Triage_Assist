"""Enhanced FastAPI app with all prediction endpoints"""
import logging
import time
import uuid
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.services.rag_pipeline import rag_pipeline
from app.services.vector_store import vector_store
from app.services.memory import conversation_memory
from app.models.schemas import (
    PatientInput, VitalSigns, ESIPredictionRequest, DeteriorationRequest,
    WaitTimeRequest, TriagePredictionResponse, SearchRequest, SearchResponse,
    HealthResponse, StatsResponse
)
from app.models.esi_predictor import esi_predictor, ESIFeatures
from app.models.deterioration_predictor import deterioration_predictor
from app.models.wait_time_predictor import wait_time_predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

start_time = time.time()


# ── Request Models ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    sources: list
    confidence: float
    latency_ms: float


# ── App Setup ──────────────────────────────────────────────────
app = FastAPI(
    title="ED Triage Assist — AI RAG + Prediction API",
    description="Emergency Department Triage: RAG knowledge base + ESI prediction + Deterioration + Wait time",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health & Stats ─────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "running", "service": "ED Triage Assist RAG + Prediction API", "version": "2.0.0"}


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        vector_store=vector_store.get_stats(),
        models={
            "esi": esi_predictor.model_version,
            "deterioration": "rules-v1.0",
            "wait_time": "rules-v1.0",
            "embeddings": settings.embedding_model,
        },
        memory=conversation_memory.get_stats(),
        uptime_seconds=round(time.time() - start_time, 1),
    )


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    return StatsResponse(
        vector_store=vector_store.get_stats(),
        memory=conversation_memory.get_stats(),
        pipeline={"total_queries": rag_pipeline.total_queries, "avg_latency_ms": round(rag_pipeline.avg_latency, 1)},
        predictions={
            "esi": esi_predictor.total_predictions,
            "deterioration": deterioration_predictor.total_predictions,
            "wait_time": wait_time_predictor.total_predictions,
        },
    )


# ── RAG Chat ───────────────────────────────────────────────────
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    try:
        result = rag_pipeline.query(
            user_query=request.query,
            session_id=request.session_id,
            use_hybrid=True,
            use_reranking=True,
        )
        return ChatResponse(
            answer=result.answer,
            sources=result.sources,
            confidence=result.confidence,
            latency_ms=result.latency_ms,
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")


# ── ESI Prediction ─────────────────────────────────────────────
@app.post("/api/predict/esi")
async def predict_esi(request: ESIPredictionRequest):
    """Predict ESI triage level for a patient."""
    try:
        features = ESIFeatures.from_patient_input(request.patient)
        prediction = esi_predictor.predict(features)

        # Augment with RAG context if requested
        rag_sources = []
        if request.use_rag_context:
            query = f"ESI classification for {request.patient.chief_complaint} age {request.patient.age}"
            rag_result = rag_pipeline.query(query, session_id="esi_prediction", top_k=3)
            prediction["reasoning"].extend([
                f"According to clinical guidelines: {rag_result.answer[:200]}"
            ])
            rag_sources = rag_result.sources

        esi_predictor.total_predictions += 1

        return {
            "esi_prediction": prediction,
            "rag_sources": rag_sources,
            "features_extracted": {
                "age": features.age,
                "hr": features.hr,
                "spo2": features.spo2,
                "rr": features.rr,
                "mental_status": features.mental_status,
                "estimated_resources": features.resource_count_estimate,
            },
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"ESI prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")


# ── Deterioration Prediction ──────────────────────────────────
@app.post("/api/predict/deterioration")
async def predict_deterioration(request: DeteriorationRequest):
    """Predict patient deterioration risk."""
    try:
        vitals = {
            "heart_rate": request.vital_signs.heart_rate,
            "blood_pressure_systolic": request.vital_signs.blood_pressure_systolic,
            "blood_pressure_diastolic": request.vital_signs.blood_pressure_diastolic,
            "spo2": request.vital_signs.spo2,
            "respiratory_rate": request.vital_signs.respiratory_rate,
            "temperature": request.vital_signs.temperature,
            "gcs": request.vital_signs.gcs,
        }
        result = deterioration_predictor.predict(
            patient_id=request.patient_id,
            vital_signs=vitals,
            current_esi=request.current_esi,
            time_in_ed_minutes=request.time_in_ed_minutes,
            chief_complaint=request.chief_complaint,
        )
        return result
    except Exception as e:
        logger.error(f"Deterioration prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")


# ── Wait Time Prediction ───────────────────────────────────────
@app.post("/api/predict/wait-time")
async def predict_wait_time(request: WaitTimeRequest):
    """Predict patient wait time."""
    try:
        patient_data = {
            "age": request.patient.age,
            "arrival_mode": request.patient.arrival_mode,
            "esi_level": 3,  # Would come from ESI prediction
        }
        ed_state = {
            "current_ed_volume": request.current_ed_volume,
            "staff_on_duty": request.staff_on_duty,
            "available_beds": request.available_beds,
            "resources_busy": request.resources_busy,
        }
        result = wait_time_predictor.predict(patient_data, ed_state)
        return result
    except Exception as e:
        logger.error(f"Wait time prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")


# ── Full Triage Prediction ────────────────────────────────────
@app.post("/api/predict/triage")
async def full_triage_prediction(request: ESIPredictionRequest):
    """Complete triage prediction: ESI + Deterioration + Wait Time."""
    try:
        t0 = time.time()

        # 1. ESI Prediction
        features = ESIFeatures.from_patient_input(request.patient)
        esi_result = esi_predictor.predict(features)

        # 2. Deterioration Prediction
        vitals = {
            "heart_rate": request.patient.vital_signs.heart_rate,
            "blood_pressure_systolic": request.patient.vital_signs.blood_pressure_systolic,
            "spo2": request.patient.vital_signs.spo2,
            "respiratory_rate": request.patient.vital_signs.respiratory_rate,
            "temperature": request.patient.vital_signs.temperature,
            "gcs": request.patient.vital_signs.gcs,
        }
        deterioration_result = deterioration_predictor.predict(
            patient_id="current",
            vital_signs=vitals,
            current_esi=esi_result["level"],
            time_in_ed_minutes=0,
            chief_complaint=request.patient.chief_complaint,
        )

        # 3. Wait Time Prediction
        wait_result = wait_time_predictor.predict(
            {"age": request.patient.age, "arrival_mode": request.patient.arrival_mode, "esi_level": esi_result["level"]},
            {"current_ed_volume": 15, "staff_on_duty": 6, "available_beds": 5, "resources_busy": {"CT": 1, "XRay": 1}},
        )

        # 4. RAG augmentation
        rag_sources = []
        if request.use_rag_context:
            try:
                rag_result = rag_pipeline.query(
                    f"Clinical guidelines for {request.patient.chief_complaint} ESI level {esi_result['level']}",
                    session_id="triage_full",
                    top_k=3,
                )
                rag_sources = rag_result.sources
            except Exception:
                pass

        latency = (time.time() - t0) * 1000

        return {
            "esi_prediction": esi_result,
            "deterioration_risk": deterioration_result,
            "wait_time": wait_result,
            "rag_sources": rag_sources,
            "processing_latency_ms": round(latency, 1),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Full triage prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")


# ── Document Search ────────────────────────────────────────────
@app.post("/api/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search clinical documents using RAG pipeline."""
    try:
        result = rag_pipeline.query(
            user_query=request.query,
            session_id=f"search_{uuid.uuid4().hex[:8]}",
            top_k=request.top_k,
            use_hybrid=request.use_hybrid,
            use_reranking=request.use_reranking,
        )
        return SearchResponse(
            query=request.query,
            results=[
                {"chunk_id": s.get("source", ""), "text": s.get("text_preview", ""),
                 "metadata": {"source": s.get("source"), "page": s.get("page"), "score": s.get("score")},
                 "score": s.get("score", 0)}
                for s in result.sources
            ],
            total_results=len(result.sources),
            latency_ms=result.latency_ms,
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(500, f"Search failed: {str(e)}")


# ── Document Ingestion ────────────────────────────────────────
@app.post("/api/ingest")
async def ingest_documents():
    """Process and ingest all documents into vector store."""
    from app.services.document_processor import DocumentProcessor
    from pathlib import Path
    try:
        docs_dir = Path(__file__).resolve().parent.parent.parent / "data"
        processor = DocumentProcessor(str(docs_dir))
        documents = processor.process_all_documents()
        all_chunks = []
        for doc in documents:
            for chunk in doc.chunks:
                all_chunks.append({
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": {
                        "source": doc.filename, "title": doc.title,
                        "file_type": doc.metadata.get("file_type", "unknown"),
                        "page": chunk.page_number, "section": chunk.section,
                        "chunk_id": chunk.chunk_id, "text": chunk.text,
                    },
                })
        if all_chunks:
            vector_store.initialize()
            vector_store.add_documents(all_chunks)
        return {"status": "complete", "documents_processed": len(documents), "total_chunks": len(all_chunks)}
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(500, f"Ingestion failed: {str(e)}")


# ── Session Management ────────────────────────────────────────
@app.get("/api/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    session = conversation_memory.get_or_create_session(session_id)
    return {
        "session_id": session_id,
        "messages": [{"role": m.role, "content": m.content[:200], "timestamp": m.timestamp}
                     for m in session.get_recent_messages(20)],
    }


@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str):
    if session_id in conversation_memory.sessions:
        del conversation_memory.sessions[session_id]
    return {"status": "cleared", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
