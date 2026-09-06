"""
Pydantic schemas for request/response validation (Flattened)
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ── Flat Patient Input Schemas ────────────────────────────────
class ESIRequest(BaseModel):
    age: int = Field(45, ge=0, le=120)
    gender: str = "unknown"
    chief_complaint: str = "Unspecified"
    bp_systolic: int = Field(120, ge=60, le=300)
    bp_diastolic: int = Field(80, ge=30, le=200)
    hr: int = Field(80, ge=20, le=300)
    rr: int = Field(16, ge=5, le=60)
    temp: float = Field(37.0, ge=32.0, le=43.0)
    spo2: float = Field(98.0, ge=50.0, le=100.0)
    medical_history: List[str] = Field(default_factory=list)
    presenting_symptoms: List[str] = Field(default_factory=list)

class DeteriorationRequest(ESIRequest):
    gcs: Optional[int] = Field(15, ge=3, le=15)
    current_medications: List[str] = Field(default_factory=list)

class WaitTimeRequest(BaseModel):
    esi_level: int = Field(..., ge=1, le=5)
    hospital_load: str = "NORMAL"
    department: str = "ED"
    day_of_week: int = 1
    hour_of_day: int = 12
    current_queue_length: int = 0
    available_doctors: int = 1
    age: int = 45
    chief_complaint: str = "Unspecified"

class TriageRequest(DeteriorationRequest):
    hospital_load: str = "NORMAL"
    department: str = "ED"
    day_of_week: int = 1
    hour_of_day: int = 12
    current_queue_length: int = 0
    available_doctors: int = 1


# ── Prediction Response Schemas ────────────────────────────────
class ESIResponse(BaseModel):
    esi_level: int
    confidence: float
    reasoning: str
    recommended_wait_time: str
    red_flags: List[str]
    protocol: str
    latency_ms: float

class DeteriorationResponse(BaseModel):
    risk_score: float
    risk_level: str
    qsofa_score: int
    deterioration_probability: float
    warning_signs: List[str]
    monitoring_recommendations: List[str]
    time_window: str
    confidence: float
    latency_ms: float

class WaitTimeResponse(BaseModel):
    predicted_wait_minutes: int
    confidence_interval_lower: int
    confidence_interval_upper: int
    factors: List[str]
    recommendation: str
    confidence: float
    latency_ms: float

class TriageResponse(BaseModel):
    esi: ESIResponse
    deterioration: DeteriorationResponse
    wait_time: WaitTimeResponse
    overall_priority: str
    latency_ms: float


# ── Document Schemas ───────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=20)
    use_hybrid: bool = True
    use_reranking: bool = True
    filters: Optional[Dict[str, Any]] = None

class SearchResult(BaseModel):
    chunk_id: Optional[str] = None
    text: Optional[str] = None
    score: float
    metadata: Optional[Dict[str, Any]] = None
    source: Optional[str] = None
    page: Optional[int] = None
    excerpt: Optional[str] = None
    section: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    latency_ms: float

class Source(BaseModel):
    source: str
    page: Optional[int] = None
    section: Optional[str] = None
    relevance: Optional[float] = None
    score: Optional[float] = None
    excerpt: Optional[str] = None
    text_preview: Optional[str] = None
    content_type: str = ""

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"
    top_k: int = Field(3, ge=1, le=10)
    use_hybrid: bool = True
    use_reranking: bool = True
    use_amer: bool = True

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: float
    latency_ms: float
    enhanced_query: Optional[str] = None
    session_id: str = "default"

class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    chunks_created: int
    message: str
    latency_ms: float

# ── System Schemas ─────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    vector_store: Dict[str, Any]
    models: Dict[str, str]
    memory: Dict[str, Any]
    uptime_seconds: float
