"""
Wait Time Prediction Model
Predicts patient wait time based on ED operational factors
"""
import logging
import math
from typing import Dict, List

logger = logging.getLogger(__name__)


class WaitTimePredictor:
    """
    Predicts ED wait time using operational model.
    Factors: ED volume, staff availability, bed capacity, patient acuity, resource constraints.
    """

    def __init__(self):
        self.total_predictions = 0
        self.error_sum = 0.0
        self.error_count = 0

    def predict(self, patient_data: dict, ed_state: dict) -> Dict:
        """Predict wait time for a patient."""
        self.total_predictions += 1

        # Extract features
        esi_level = patient_data.get("esi_level", 3)
        age = patient_data.get("age", 40)
        arrival_mode = patient_data.get("arrival_mode", "walk-in")
        ed_volume = ed_state.get("current_ed_volume", 20)
        staff_count = ed_state.get("staff_on_duty", 6)
        available_beds = ed_state.get("available_beds", 10)
        busy_resources = ed_state.get("resources_busy", {})

        # ── Base wait time by ESI level ──
        base_waits = {1: 0, 2: 10, 3: 25, 4: 45, 5: 60}
        base_wait = base_waits.get(esi_level, 30)

        # ── Congestion factor ──
        # Optimal ratio: 1 staff per 3-4 patients
        congestion_ratio = ed_volume / max(staff_count, 1)
        congestion_multiplier = 1.0 + max(0, (congestion_ratio - 3.0) * 0.4)

        # ── Bed availability factor ──
        bed_factor = 1.0 + max(0, (ed_volume - available_beds) * 0.05)

        # ── Resource queue factor ──
        resource_penalty = 0
        for resource, count in busy_resources.items():
            if count > 0:
                resource_penalty += count * 3  # 3 min per queued resource

        # ── Arrival mode factor ──
        arrival_factor = 1.0 if arrival_mode == "ambulance" else 1.1

        # ── Age factor (pediatric + elderly may need more time) ──
        age_factor = 1.0
        if age < 5 or age > 75:
            age_factor = 1.1

        # ── Time of day factor ──
        hour = __import__('datetime').datetime.now().hour
        if 10 <= hour <= 20:  # Peak hours
            time_factor = 1.3
        elif 20 <= hour <= 23 or 6 <= hour <= 9:
            time_factor = 1.1
        else:
            time_factor = 0.9

        # ── Calculate ──
        estimated_wait = base_wait * congestion_multiplier * bed_factor * arrival_factor * age_factor * time_factor
        estimated_wait += resource_penalty
        estimated_wait = max(0, math.ceil(estimated_wait))

        # Confidence based on data completeness
        confidence = 0.8 if ed_state.get("current_ed_volume") else 0.5

        # Queue position
        queue_position = max(1, math.ceil(ed_volume * (6 - esi_level) / 15))

        # Factors
        factors = []
        if congestion_multiplier > 1.2:
            factors.append(f"High congestion ratio: {congestion_ratio:.1f}")
        if bed_factor > 1.0:
            factors.append(f"Limited bed availability: {available_beds} free")
        if resource_penalty > 0:
            factors.append(f"Resource queues: +{resource_penalty} min")
        if time_factor > 1.1:
            factors.append("Peak hour demand")

        # Recommendation
        if estimated_wait > 60:
            recommendation = "Consider redirecting non-urgent patients. Alert additional staff."
        elif estimated_wait > 45:
            recommendation = "High wait expected. Consider opening additional treatment bays."
        elif estimated_wait > 30:
            recommendation = "Moderate wait. Monitor ED flow and resource utilization."
        else:
            recommendation = "Normal wait times expected. Maintain current staffing."

        return {
            "estimated_wait_minutes": estimated_wait,
            "confidence": confidence,
            "queue_position": queue_position,
            "factors": factors,
            "recommendation": recommendation,
        }


wait_time_predictor = WaitTimePredictor()
