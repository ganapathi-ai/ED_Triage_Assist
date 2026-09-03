"""
Deterioration Prediction Model
Predicts patient deterioration risk using clinical rules + time-series analysis
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class VitalTrend:
    """Tracks vital sign trends over time."""

    def __init__(self, max_points: int = 10):
        self.hr: deque = deque(maxlen=max_points)
        self.bp_systolic: deque = deque(maxlen=max_points)
        self.bp_diastolic: deque = deque(maxlen=max_points)
        self.spo2: deque = deque(maxlen=max_points)
        self.rr: deque = deque(maxlen=max_points)
        self.temp: deque = deque(maxlen=max_points)
        self.timestamps: deque = deque(maxlen=max_points)

    def add(self, vitals: dict):
        now = datetime.now()
        self.hr.append(vitals.get("heart_rate", 0))
        self.bp_systolic.append(vitals.get("blood_pressure_systolic", 0))
        self.bp_diastolic.append(vitals.get("blood_pressure_diastolic", 0))
        self.spo2.append(vitals.get("spo2", 100))
        self.rr.append(vitals.get("respiratory_rate", 0))
        self.temp.append(vitals.get("temperature", 37.0))
        self.timestamps.append(now)

    def get_trend(self, vital: str) -> str:
        """Get trend direction for a vital sign."""
        values = list(getattr(self, vital, []))
        if len(values) < 2:
            return "stable"
        recent = sum(values[-3:]) / min(len(values[-3:]), len(values))
        earlier = sum(values[:3]) / min(len(values[:3]), len(values))
        if recent > earlier * 1.1:
            return "deteriorating"
        elif recent < earlier * 0.9:
            return "improving"
        return "stable"

    def get_variability(self, vital: str) -> float:
        """Get coefficient of variation for a vital sign."""
        values = list(getattr(self, vital, []))
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        return std / mean if mean > 0 else 0.0


class DeteriorationPredictor:
    """
    Predicts patient deterioration risk based on:
    - Current vital signs vs age-specific thresholds
    - Vital sign trends (derivative analysis)
    - ESI level and time in department
    - Clinical presentation patterns
    """

    # Risk thresholds
    CRITICAL_SPO2 = 88
    CRITICAL_SBP = 80
    CRITICAL_HR_LOW = 50
    CRITICAL_HR_HIGH = 140
    CRITICAL_RR_LOW = 8
    CRITICAL_RR_HIGH = 35
    SEPSIS_THRESHOLD_QSOFA = 2  # qSOFA score

    def __init__(self):
        self.patient_trends: Dict[str, VitalTrend] = {}
        self.total_predictions = 0

    def predict(self, patient_id: str, vital_signs: dict, current_esi: int,
                time_in_ed_minutes: int, chief_complaint: str) -> Dict:
        """Predict deterioration risk."""
        self.total_predictions += 1
        risk_score = 0.0
        risk_factors = []
        warning_signs = []
        recommended_actions = []

        # Initialize trend tracker
        if patient_id not in self.patient_trends:
            self.patient_trends[patient_id] = VitalTrend()
        trend = self.patient_trends[patient_id]
        trend.add(vital_signs)

        # ── Current Vitals Risk ──
        spo2 = vital_signs.get("spo2", 100)
        sbp = vital_signs.get("blood_pressure_systolic", 120)
        hr = vital_signs.get("heart_rate", 80)
        rr = vital_signs.get("respiratory_rate", 16)
        temp = vital_signs.get("temperature", 37.0)

        if spo2 < self.CRITICAL_SPO2:
            risk_score += 0.3
            warning_signs.append(f"Critical hypoxemia: SpO2 {spo2}%")
            recommended_actions.append("Immediate oxygen supplementation")

        if sbp < self.CRITICAL_SBP:
            risk_score += 0.3
            warning_signs.append(f"Severe hypotension: SBP {sbp}")
            recommended_actions.append("Fluid resuscitation, consider vasopressors")

        if rr < self.CRITICAL_RR_LOW or rr > self.CRITICAL_RR_HIGH:
            risk_score += 0.15
            warning_signs.append(f"Critical respiratory rate: {rr}/min")

        if hr > self.CRITICAL_HR_HIGH or hr < self.CRITICAL_HR_LOW:
            risk_score += 0.1
            warning_signs.append(f"Concerning heart rate: {hr} bpm")

        # ── qSOFA Score ──
        qsofa = 0
        if hr >= 100:
            qsofa += 1
        if sbp <= 100:
            qsofa += 1
        altered = vital_signs.get("gcs", 15) < 15
        if altered:
            qsofa += 1

        if qsofa >= self.SEPSIS_THRESHOLD_QSOFA:
            risk_score += 0.2
            warning_signs.append(f"qSOFA score {qsofa}/3 — sepsis risk")
            recommended_actions.append("Blood cultures, lactate, broad-spectrum antibiotics")

        # ── Trend Analysis ──
        spo2_trend = trend.get_trend("spo2")
        sbp_trend = trend.get_trend("bp_systolic")
        rr_trend = trend.get_trend("rr")

        if spo2_trend == "deteriorating":
            risk_score += 0.1
            warning_signs.append("SpO2 trending downward")
        if sbp_trend == "deteriorating":
            risk_score += 0.1
            warning_signs.append("Blood pressure trending downward")
        if rr_trend == "deteriorating":
            risk_score += 0.1
            warning_signs.append("Respiratory rate worsening")

        # ── ESI Level Adjustment ──
        if current_esi == 1:
            risk_score += 0.2
        elif current_esi == 2:
            risk_score += 0.1

        # ── Time in ED ──
        if time_in_ed_minutes > 60 and current_esi <= 3:
            risk_score += 0.1
            warning_signs.append(f"Extended ED stay: {time_in_ed_minutes} minutes")

        # ── Chief Complaint Patterns ──
        high_risk_complaints = {
            "chest pain": 0.15, "shortness of breath": 0.15, "sepsis": 0.25,
            "stroke": 0.15, "altered mental status": 0.15, "trauma": 0.1,
            "overdose": 0.1, "anaphylaxis": 0.2, "shock": 0.2,
        }
        for keyword, risk_add in high_risk_complaints.items():
            if keyword in chief_complaint.lower():
                risk_score += risk_add
                break

        # ── Determine risk level ──
        risk_score = min(risk_score, 1.0)
        if risk_score >= 0.6:
            risk_level = "Critical"
            recommended_actions.extend(["Continuous monitoring", "Escalate to attending immediately", "Consider ICU admission"])
        elif risk_score >= 0.4:
            risk_level = "High"
            recommended_actions.extend(["Frequent reassessment every 15 min", "Update care team"])
        elif risk_score >= 0.2:
            risk_level = "Medium"
            recommended_actions.extend(["Reassess every 30 min", "Monitor trends"])
        else:
            risk_level = "Low"
            recommended_actions.append("Standard monitoring")

        return {
            "risk_score": round(risk_score, 3),
            "risk_level": risk_level,
            "warning_signs": list(set(warning_signs)),
            "recommended_actions": list(set(recommended_actions)),
            "qsofa_score": qsofa,
            "vital_trends": {
                "spo2": spo2_trend,
                "bp_systolic": sbp_trend,
                "rr": rr_trend,
            },
            "time_to_reassess_minutes": self._reassess_interval(risk_level),
            "predicted_conditions": self._predict_conditions(chief_complaint, vital_signs, risk_score),
        }

    def _reassess_interval(self, risk_level: str) -> int:
        intervals = {"Critical": 5, "High": 15, "Medium": 30, "Low": 60}
        return intervals.get(risk_level, 30)

    def _predict_conditions(self, complaint: str, vitals: dict, risk: float) -> List[Dict]:
        """Predict possible conditions based on presentation."""
        conditions = []
        cc = complaint.lower()

        if "chest pain" in cc:
            conditions.append({"condition": "Acute Coronary Syndrome", "probability": 0.3 + risk * 0.3})
            conditions.append({"condition": "Pulmonary Embolism", "probability": 0.15})
        if "shortness of breath" in cc or "dyspnea" in cc:
            conditions.append({"condition": "COPD/Asthma Exacerbation", "probability": 0.3})
            conditions.append({"condition": "Pneumonia", "probability": 0.2})
            conditions.append({"condition": "Heart Failure", "probability": 0.15})
        if "abdominal pain" in cc:
            conditions.append({"condition": "Appendicitis", "probability": 0.25})
            conditions.append({"condition": "Cholecystitis", "probability": 0.15})
        if "headache" in cc:
            conditions.append({"condition": "Migraine", "probability": 0.3})
            conditions.append({"condition": "Subarachnoid Hemorrhage", "probability": 0.1})
        if "fever" in cc or "temperature" in cc:
            conditions.append({"condition": "Sepsis", "probability": 0.2 + risk * 0.3})
            conditions.append({"condition": "Viral Syndrome", "probability": 0.3})

        return sorted(conditions, key=lambda x: x["probability"], reverse=True)[:5]


deterioration_predictor = DeteriorationPredictor()
