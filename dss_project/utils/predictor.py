"""Prediction wrapper and explanation logic."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

FEATURE_MAPPING = {
    "stress_score": "Stress_Score",
    "anxiety_score": "Anxiety_Score",
    "exam_pressure": "Exam_Pressure",
    "sleep_hours": "Sleep_Hours",
    "social_support": "Social_Support",
    "heart_rate": "Heart_Rate",
    "physical_activity": "Physical_Activity",
    "assignment_load": "Assignment_Load",
    "study_hours": "Study_Hours",
    "attendance": "Attendance",
    "screen_time": "Screen_Time",
    "facial_emotion": "Facial_Emotion",
    "mood_state": "Mood_State",
    "intervention_response": "Intervention_Response",
    "reward_score": "Reward_Score",
}

TARGETS = [
    "stress_level",
    "anxiety_level",
    "final_state",
    "recommended_intervention",
]


@dataclass
class PredictionOutput:
    success: bool
    predictions: Dict[str, str]
    confidence: float
    explanation: str


class Predictor:
    """Wrapper for DSS model predictions."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.pipeline = self._load_model()
        self.is_ready = self.pipeline is not None

    def _load_model(self) -> Any:
        try:
            return joblib.load(self.model_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to load model: %s", exc)
            return None

    def _prepare_frame(self, payload: Dict[str, Any]) -> pd.DataFrame:
        mapped = {FEATURE_MAPPING[key]: payload[key] for key in FEATURE_MAPPING}
        return pd.DataFrame([mapped])

    def _confidence(self, probas: List[np.ndarray]) -> float:
        max_probs = [float(np.max(prob)) for prob in probas if prob.size]
        if not max_probs:
            return 0.0
        return float(np.mean(max_probs))

    def _explanation(self, payload: Dict[str, Any], recommendation: str) -> str:
        reasons = []
        if payload["stress_score"] >= 60:
            reasons.append(f"elevated stress score ({payload['stress_score']})")
        if payload["sleep_hours"] <= 6:
            reasons.append(f"insufficient sleep ({payload['sleep_hours']} hours)")
        if payload["exam_pressure"] >= 7:
            reasons.append(f"high exam pressure ({payload['exam_pressure']})")

        if reasons:
            cause = " and ".join(reasons)
            return f"High stress detected due to {cause}. {recommendation} recommended."

        return f"Stress indicators are within expected ranges. {recommendation} recommended."

    def _apply_post_rules(
        self, predictions: Dict[str, str], payload: Dict[str, Any]
    ) -> Dict[str, str]:
        stress_score = float(payload["stress_score"])
        anxiety_score = float(payload["anxiety_score"])

        if stress_score >= 65:
            predictions["stress_level"] = "High"
        elif stress_score >= 35:
            predictions["stress_level"] = "Medium"
        else:
            predictions["stress_level"] = "Low"

        if anxiety_score >= 65:
            predictions["anxiety_level"] = "High"
        elif anxiety_score >= 35:
            predictions["anxiety_level"] = "Medium"
        else:
            predictions["anxiety_level"] = "Low"

        return predictions

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.pipeline:
            raise RuntimeError("Model is not loaded")

        frame = self._prepare_frame(payload)
        prediction = self.pipeline.predict(frame)[0]

        probas = self.pipeline.predict_proba(frame)
        confidence = self._confidence(probas)

        predictions = dict(zip(TARGETS, prediction))
        predictions = self._apply_post_rules(predictions, payload)
        recommendation = predictions["recommended_intervention"]
        explanation = self._explanation(payload, recommendation)

        output = PredictionOutput(
            success=True,
            predictions=predictions,
            confidence=round(confidence, 2),
            explanation=explanation,
        )
        return output.__dict__
