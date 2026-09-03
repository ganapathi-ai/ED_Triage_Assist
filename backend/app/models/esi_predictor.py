"""
ESI Triage Prediction Model
Rule-based + ML-ready model for Emergency Severity Index classification
"""
import logging
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ESIFeatures:
    """Extracted features for ESI prediction."""
    age: int
    gender: str
    hr: int
    bp_systolic: int
    bp_diastolic: int
    spo2: float
    rr: int
    temp: float
    glucose: Optional[float]
    gcs: Optional[int]
    mental_status: str
    pain_score: Optional[int]
    chief_complaint: str
    mechanism_of_injury: Optional[str]
    has_allergy: bool
    resource_count_estimate: int

    @classmethod
    def from_patient_input(cls, patient) -> "ESIFeatures":
        vs = patient.vital_signs
        cc_lower = patient.chief_complaint.lower()

        # Estimate resource count from complaint
        resource_count = cls._estimate_resources(patient)

        return cls(
            age=patient.age,
            gender=patient.gender,
            hr=vs.heart_rate,
            bp_systolic=vs.blood_pressure_systolic,
            bp_diastolic=vs.blood_pressure_diastolic,
            spo2=vs.spo2,
            rr=vs.respiratory_rate,
            temp=vs.temperature,
            glucose=vs.glucose,
            gcs=vs.gcs,
            mental_status=patient.mental_status.lower(),
            pain_score=patient.pain_score,
            chief_complaint=cc_lower,
            mechanism_of_injury=patient.mechanism_of_injury,
            has_allergy=len(patient.allergies) > 0,
            resource_count_estimate=resource_count,
        )

    @staticmethod
    def _estimate_resources(patient) -> int:
        """Estimate number of resources needed based on complaint."""
        cc = patient.chief_complaint.lower()
        high_resource_keywords = ["fracture", "laceration", "burn", "trauma", "abdominal pain",
                                   "chest pain", "shortness of breath", "stroke", "sepsis"]
        med_resource_keywords = ["migraine", "infection", "fever", "UTI", "pneumonia",
                                  "asthma", "COPD", "cellulitis"]
        low_resource_keywords = ["rash", "prescription", "follow-up", "sore throat",
                                  "ear pain", "ankle", "sprain"]

        if any(k in cc for k in high_resource_keywords):
            return 3  # Imaging + labs + consult likely
        elif any(k in cc for k in med_resource_keywords):
            return 2  # Labs + treatment
        elif any(k in cc for k in low_resource_keywords):
            return 1  # Simple treatment or prescription
        return 2  # Default


class ESIPredictor:
    """
    ESI Triage Level Predictor.
    Implements the ESI v5 algorithm with clinical rules.
    Can be extended with ML models (XGBoost, Neural Network).
    """

    # Age-specific vital sign thresholds (from ESI Handbook)
    AGE_VITAL_THRESHOLDS = {
        "infant": {"hr": (100, 160), "rr": (30, 60), "bp_systolic": (70, 100)},
        "toddler": {"hr": (80, 140), "rr": (20, 40), "bp_systolic": (75, 100)},
        "child": {"hr": (70, 120), "rr": (15, 30), "bp_systolic": (80, 110)},
        "adolescent": {"hr": (60, 100), "rr": (12, 20), "bp_systolic": (90, 120)},
        "adult": {"hr": (60, 100), "rr": (12, 20), "bp_systolic": (90, 140)},
        "elderly": {"hr": (60, 100), "rr": (12, 20), "bp_systolic": (100, 160)},
    }

    def __init__(self):
        self.model_version = "rules-v5.0"
        self.total_predictions = 0
        self.accuracy_tracking: List[Tuple[int, int]] = []  # (predicted, actual)

    def predict(self, features: ESIFeatures) -> Dict:
        """
        Predict ESI level using the 4-decision-point algorithm.
        Decision Point A → B → C → D
        """
        reasoning = []
        confidence = 0.85

        # ── Decision Point A: Immediate Lifesaving Intervention? ──
        dp_a_result = self._decision_point_a(features, reasoning)
        if dp_a_result is not None:
            return self._build_result(1, "Resuscitation", confidence + 0.1, reasoning,
                                       ["Immediate resuscitation", "Call attending physician", "Prepare for ICU"])

        # ── Decision Point B: High-Risk / Confused / Severe Pain? ──
        dp_b_result = self._decision_point_b(features, reasoning)
        if dp_b_result is not None:
            level = dp_b_result
            return self._build_result(level, self._level_name(level), confidence, reasoning,
                                       self._recommended_actions(level))

        # ── Decision Point C: Resource Count ──
        dp_c_result = self._decision_point_c(features, reasoning)
        if dp_c_result is not None:
            return self._build_result(dp_c_result, self._level_name(dp_c_result), confidence, reasoning,
                                       self._recommended_actions(dp_c_result))

        # ── Decision Point D: Vital Signs ──
        dp_d_result = self._decision_point_d(features, reasoning)
        if dp_d_result is not None:
            return self._build_result(dp_d_result, self._level_name(dp_d_result), confidence - 0.05, reasoning,
                                       self._recommended_actions(dp_d_result))

        # Default
        return self._build_result(5, "Non-Urgent", confidence - 0.1, reasoning + ["No specific criteria met"],
                                   ["Discharge with follow-up"])

    def _decision_point_a(self, f: ESIFeatures, reasoning: List[str]) -> Optional[int]:
        """Check if patient needs immediate lifesaving intervention."""
        # Airway compromise
        if "stridor" in f.chief_complaint or "choking" in f.chief_complaint or "can't breathe" in f.chief_complaint:
            reasoning.append("Airway compromise detected — immediate intervention required")
            return 1

        # Respiratory arrest / failure
        if f.rr < 8 or f.spo2 < 85:
            reasoning.append(f"Respiratory failure: RR={f.rr}, SpO2={f.spo2}")
            return 1

        # Hemodynamic instability
        if f.bp_systolic < 70:
            reasoning.append(f"Severe hypotension: SBP={f.bp_systolic}")
            return 1

        # Cardiac arrest indicators
        if "cardiac arrest" in f.chief_complaint or "no pulse" in f.chief_complaint:
            reasoning.append("Cardiac arrest presentation")
            return 1

        # Unresponsive
        if f.mental_status == "unresponsive" and f.gcs is not None and f.gcs < 8:
            reasoning.append(f"GCS {f.gcs} with unresponsive — airway at risk")
            return 1

        return None

    def _decision_point_b(self, f: ESIFeatures, reasoning: List[str]) -> Optional[int]:
        """Check high-risk situations, confused/lethargic, severe pain."""
        # High-risk chief complaints
        high_risk = {
            "chest pain": "Chest pain — high-risk presentation",
            "severe headache": "Severe headache — possible CVA",
            "sudden vision loss": "Vision loss — neuro emergency",
            "paralysis": "Paralysis — possible stroke",
            "severe abdominal pain": "Severe abdominal pain — surgical emergency",
            "suicidal": "Psychiatric emergency",
            "overdose": "Overdose — toxicology emergency",
            "anaphylaxis": "Anaphylaxis — immediate threat",
            "shock": "Shock — hemodynamic emergency",
            "sepsis": "Sepsis — time-critical",
            "altered mental status": "Altered mental status — high risk",
            "fall.*head": "Head injury with fall mechanism",
            "trauma.*head": "Head trauma",
        }

        for keyword, reason in high_risk.items():
            if re.search(keyword, f.chief_complaint):
                reasoning.append(reason)
                return 2

        # Confused / lethargic / disoriented
        if f.mental_status in ["confused", "lethargic", "disoriented"]:
            reasoning.append(f"Altered mental status: {f.mental_status}")
            return 2

        # High-risk vital signs
        thresholds = self._get_age_thresholds(f.age)
        hr_high = f.hr > thresholds["hr"][1] * 1.5
        rr_high = f.rr > thresholds["rr"][1] * 1.5
        bp_low = f.bp_systolic < thresholds["bp_systolic"][0]
        bp_high = f.bp_systolic > thresholds["bp_systolic"][1] * 1.5

        if f.temp > 39.5:
            reasoning.append(f"High fever: {f.temp}°C")
            return 2

        if f.spo2 < 90 and f.spo2 >= 85:
            reasoning.append(f"Hypoxemia: SpO2 {f.spo2}%")
            return 2

        # Severe pain
        if f.pain_score is not None and f.pain_score >= 8:
            reasoning.append(f"Severe pain: score {f.pain_score}/10")
            return 2

        if hr_high or rr_high or bp_low or bp_high:
            abnormal = []
            if hr_high: abnormal.append(f"HR {f.hr}")
            if rr_high: abnormal.append(f"RR {f.rr}")
            if bp_low: abnormal.append(f"SBP {f.bp_systolic}")
            if bp_high: abnormal.append(f"SBP {f.bp_systolic}")
            reasoning.append(f"Abnormal vitals: {', '.join(abnormal)}")
            return 2

        return None

    def _decision_point_c(self, f: ESIFeatures, reasoning: List[str]) -> Optional[int]:
        """Determine ESI level based on resource count."""
        n = f.resource_count_estimate
        if n >= 2:
            reasoning.append(f"Estimated {n} resources needed")
            return 3
        elif n == 1:
            reasoning.append("Estimated 1 resource needed")
            return 4
        else:
            reasoning.append("No resources needed beyond exam")
            return 5

    def _decision_point_d(self, f: ESIFeatures, reasoning: List[str]) -> Optional[int]:
        """Check abnormal vital signs (Decision Point D)."""
        thresholds = self._get_age_thresholds(f.age)
        abnormal_vitals = []

        if not (thresholds["hr"][0] <= f.hr <= thresholds["hr"][1]):
            abnormal_vitals.append(f"HR {f.hr} (normal: {thresholds['hr'][0]}-{thresholds['hr'][1]})")
        if not (thresholds["rr"][0] <= f.rr <= thresholds["rr"][1]):
            abnormal_vitals.append(f"RR {f.rr} (normal: {thresholds['rr'][0]}-{thresholds['rr'][1]})")
        if f.spo2 < 92:
            abnormal_vitals.append(f"SpO2 {f.spo2}% (< 92%)")
        if f.temp >= 38.5:
            abnormal_vitals.append(f"Temp {f.temp}°C")

        if abnormal_vitals:
            reasoning.append(f"Abnormal vitals (DP-D): {'; '.join(abnormal_vitals)}")
            return 3

        return None

    def _get_age_thresholds(self, age: int) -> Dict:
        """Get age-specific vital sign thresholds."""
        if age < 1:
            return self.AGE_VITAL_THRESHOLDS["infant"]
        elif age < 3:
            return self.AGE_VITAL_THRESHOLDS["toddler"]
        elif age < 12:
            return self.AGE_VITAL_THRESHOLDS["child"]
        elif age < 18:
            return self.AGE_VITAL_THRESHOLDS["adolescent"]
        elif age > 65:
            return self.AGE_VITAL_THRESHOLDS["elderly"]
        else:
            return self.AGE_VITAL_THRESHOLDS["adult"]

    def _build_result(self, level: int, name: str, confidence: float, reasoning: List[str], actions: List[str]) -> Dict:
        return {
            "level": level,
            "level_name": name,
            "confidence": min(confidence, 0.99),
            "reasoning": reasoning,
            "recommended_actions": actions,
            "resources_needed": self._get_resources_for_level(level),
            "discrepancy_flag": False,
        }

    def _level_name(self, level: int) -> str:
        return {1: "Resuscitation", 2: "Emergent", 3: "Urgent", 4: "Less Urgent", 5: "Non-Urgent"}.get(level, "Unknown")

    def _recommended_actions(self, level: int) -> List[str]:
        actions = {
            1: ["Immediate resuscitation", "Call attending NOW", "Activate trauma/resuscitation team", "Prepare ICU"],
            2: ["Urgent assessment within 15 min", "Continuous monitoring", "Consider ICU consultation"],
            3: ["Standard assessment", "Diagnostic workup", "Reassess if condition changes"],
            4: ["Standard assessment", "Single resource intervention"],
            5: ["Brief assessment", "Discharge or minimal intervention"],
        }
        return actions.get(level, ["Clinical assessment"])

    def _get_resources_for_level(self, level: int) -> List[str]:
        resources = {
            1: ["Labs", "Imaging", "IV fluids", "IV medications", "Specialist consult", "Procedures"],
            2: ["Labs", "Imaging", "IV fluids", "Specialist consult"],
            3: ["Labs", "Imaging", "IV fluids"],
            4: ["Single resource (labs or imaging)"],
            5: ["No additional resources"],
        }
        return resources.get(level, [])


# Global instance
esi_predictor = ESIPredictor()
