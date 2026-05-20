"""Prediction wrapper and explanation logic."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import export_graphviz

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

CATEGORICAL_COLUMNS = [
    "Facial_Emotion",
    "Mood_State",
    "Intervention_Response",
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
        self._tree_cache: Dict[str, str] | None = None
        self._feature_names_cache: List[str] | None = None

    def _load_model(self) -> Any:
        try:
            return joblib.load(self.model_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to load model: %s", exc)
            return None

    def _prepare_frame(self, payload: Dict[str, Any]) -> pd.DataFrame:
        mapped = {FEATURE_MAPPING[key]: payload[key] for key in FEATURE_MAPPING}
        return pd.DataFrame([mapped])

    def _get_preprocessor(self) -> Any:
        return self.pipeline.named_steps["preprocessor"]

    def _get_feature_names(self) -> List[str]:
        if self._feature_names_cache is not None:
            return self._feature_names_cache

        preprocessor = self._get_preprocessor()
        input_features = list(preprocessor.feature_names_in_)
        raw_names = preprocessor.get_feature_names_out(input_features)
        formatted = [self._format_feature_label(name) for name in raw_names]
        self._feature_names_cache = list(formatted)
        return self._feature_names_cache

    def _format_feature_label(self, raw_name: str) -> str:
        name = raw_name.split("__", 1)[-1]
        for column in CATEGORICAL_COLUMNS:
            if name.startswith(f"{column}_"):
                category = name[len(column) + 1 :]
                return f"{column}={category}"
        return name

    def _split_onehot(self, feature_name: str) -> Tuple[str, str] | None:
        for column in CATEGORICAL_COLUMNS:
            prefix = f"{column}="
            if feature_name.startswith(prefix):
                return column, feature_name[len(prefix) :]
        return None

    def _format_decision_step(
        self,
        feature_name: str,
        value: float,
        threshold: float,
        is_left: bool,
    ) -> str:
        direction = "left" if is_left else "right"
        onehot = self._split_onehot(feature_name)
        if onehot:
            column, category = onehot
            is_match = value > 0.5
            result = "true" if is_match else "false"
            return f"{column} == {category} is {result} -> {direction}"

        return (
            f"{feature_name} <= {threshold:.2f} "
            f"(value={value:.2f}) -> {direction}"
        )

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

    def _build_decision_path(
        self, estimator: Any, feature_names: List[str], sample: np.ndarray
    ) -> List[str]:
        tree = estimator.tree_
        node = 0
        steps: List[str] = []
        while tree.feature[node] != -2:
            feature_index = tree.feature[node]
            threshold = tree.threshold[node]
            value = float(sample[feature_index])
            is_left = value <= threshold
            feature_name = feature_names[feature_index]
            steps.append(
                self._format_decision_step(feature_name, value, threshold, is_left)
            )
            node = tree.children_left[node] if is_left else tree.children_right[node]
        return steps

    def get_tree_dots(self) -> Dict[str, str]:
        if self._tree_cache is not None:
            return self._tree_cache

        if not self.pipeline:
            return {}

        feature_names = self._get_feature_names()
        estimators = self.pipeline.named_steps["model"].estimators_
        trees: Dict[str, str] = {}
        for target, estimator in zip(TARGETS, estimators):
            class_names = [str(label) for label in estimator.classes_]
            dot = export_graphviz(
                estimator,
                feature_names=feature_names,
                class_names=class_names,
                filled=True,
                rounded=True,
                proportion=False,
                impurity=False,
            )
            trees[target] = dot

        self._tree_cache = trees
        return trees

    def get_decision_paths(self, payload: Dict[str, Any]) -> Dict[str, List[str]]:
        if not self.pipeline:
            return {}

        frame = self._prepare_frame(payload)
        preprocessor = self._get_preprocessor()
        transformed = preprocessor.transform(frame)
        sample = np.asarray(transformed)[0]
        feature_names = self._get_feature_names()

        estimators = self.pipeline.named_steps["model"].estimators_
        paths: Dict[str, List[str]] = {}
        for target, estimator in zip(TARGETS, estimators):
            paths[target] = self._build_decision_path(
                estimator, feature_names, sample
            )
        return paths

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
        decision_paths = self.get_decision_paths(payload)

        output = PredictionOutput(
            success=True,
            predictions=predictions,
            confidence=round(confidence, 2),
            explanation=explanation,
        )
        response = output.__dict__
        response["decision_paths"] = decision_paths
        return response
