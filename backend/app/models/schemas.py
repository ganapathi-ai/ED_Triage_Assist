"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ── Patient Input Schemas ──────────────────────────────────────
class VitalSigns(BaseModel):
    heart_rate: int = Field(..., ge=20, le=300, description="Heart rate in bpm")
    blood_pressure_systolic: int = Field(..., ge=60, le=300, description="Systolic BP")
    blood_pressure_diastolic: int = Field(..., ge=30, le=200, description="Diastolic BP")
    spo2: float = Field(..., ge=50, le=100, description="Oxygen saturation %")
    respiratory_rate: int = Field(..., ge=5, le=60, description="Respiratory rate /min")
    temperature: float = Field(..., ge=32.0, le=43.0, description="Temperature in Celsius")
    glucose: Optional[float] = Field(None, ge=20, le=800, description="Blood glucose mg/dL")
    gcs: Optional[int] = Field(None, ge=3, le=15, description="Glasgow Coma Scale")


class PatientInput(BaseModel):
    """Input schema for patient triage prediction."""
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    gender: str = Field(..., description="Patient gender")
    chief_complaint: str = Field(..., description="Primary complaint / presenting symptoms")
    vital_signs: VitalSigns
    mental_status: str = Field(..., description="Alert / Confused / Lethargic / Unresponsive")
    pain_score: Optional[int] = Field(None, ge=0, le=10, description="Pain score 0-10")
    allergies: List[str] = Field(default_factory=list, description="Known allergies")
    current_medications: List[str] = Field(default_factory=list, description="Current medications")
    past_medical_history: List[str] = Field(default_factory=list, description="Relevant PMH")
    mechanism_of_injury: Optional[str] = Field(None, description="If trauma, mechanism")
    arrival_mode: str = Field("walk-in", description="Walk-in / Ambulance / Transfer")
    triage_note: Optional[str] = Field(None, description="Additional triage notes")


class ESIPredictionRequest(BaseModel):
    """Request for ESI classification prediction."""
    patient: PatientInput
    use_rag_context: bool = Field(True, description="Use RAG to augment prediction")


class DeteriorationRequest(BaseModel):
    """Request for patient deterioration prediction."""
    patient_id: str
    vital_signs: VitalSigns
    current_esi: int = Field(..., ge=1, le=5)
    time_in_ed_minutes: int = Field(..., ge=0, description="Time since ED arrival")
    chief_complaint: str


class WaitTimeRequest(BaseModel):
    """Request for wait time prediction."""
    patient: PatientInput
    current_ed_volume: int = Field(..., description="Current patients in ED")
    staff_on_duty: int = Field(..., description="Number of staff currently on duty")
    available_beds: int = Field(..., description="Available ED beds")
    resources_busy: Dict[str, int] = Field(default_factory=dict, description="Busy resources")


# ── Prediction Response Schemas ────────────────────────────────
class ESILevelResult(BaseModel):
    level: int = Field(..., ge=1, le=5, description="Predicted ESI level")
    level_name: str = Field(..., description="ESI level name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence")
    reasoning: List[str] = Field(default_factory=list, description="Clinical reasoning")
    discrepancy_flag: bool = Field(False, description="Flag if differs significantly from typical")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended next steps")
    resources_needed: List[str] = Field(default_factory=list, description="Estimated resources")


class DeteriorationResult(BaseModel):
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Deterioration risk 0-1")
    risk_level: str = Field(..., description="Low / Medium / High / Critical")
    predicted_conditions: List[Dict[str, Any]] = Field(default_factory=list)
    warning_signs: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    time_to_reassess_minutes: int = Field(..., description="Suggested reassessment interval")


class WaitTimeResult(BaseModel):
    estimated_wait_minutes: int = Field(..., description="Estimated wait in minutes")
    confidence: float = Field(..., ge=0.0, le=1.0)
    queue_position: int = Field(..., description="Position in queue")
    factors: List[str] = Field(default_factory=list, description="Factors affecting wait")
    recommendation: str = Field(..., description="Operational recommendation")


class TriagePredictionResponse(BaseModel):
    """Complete triage prediction response."""
    esi_prediction: ESILevelResult
    deterioration_risk: DeteriorationResult
    wait_time: WaitTimeResult
    rag_sources: List[Dict[str, Any]] = Field(default_factory=list)
    processing_latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.now)


# ── Document Schemas ───────────────────────────────────────────
class DocumentChunkSchema(BaseModel):
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    score: Optional[float] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=20)
    use_hybrid: bool = True
    use_reranking: bool = True
    filters: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    query: str
    results: List[DocumentChunkSchema]
    total_results: int
    latency_ms: float


# ── System Schemas ─────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    vector_store: Dict[str, Any]
    models: Dict[str, str]
    memory: Dict[str, Any]
    uptime_seconds: float


class StatsResponse(BaseModel):
    vector_store: Dict[str, Any]
    memory: Dict[str, Any]
    pipeline: Dict[str, Any]
    predictions: Dict[str, int]
