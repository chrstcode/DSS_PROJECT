"""Model training script for DSS stress prediction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

LOGGER = logging.getLogger(__name__)

FEATURES: Dict[str, List[str]] = {
    "tier_1": ["Stress_Score", "Anxiety_Score", "Exam_Pressure"],
    "tier_2": [
        "Sleep_Hours",
        "Social_Support",
        "Heart_Rate",
        "Physical_Activity",
    ],
    "tier_3": [
        "Assignment_Load",
        "Study_Hours",
        "Attendance",
        "Screen_Time",
    ],
    "tier_4": [
        "Facial_Emotion",
        "Mood_State",
        "Intervention_Response",
        "Reward_Score",
    ],
}

CATEGORICAL_FEATURES = [
    "Facial_Emotion",
    "Mood_State",
    "Intervention_Response",
]

TARGETS = [
    "Stress_Level",
    "Anxiety_Level",
    "Final_State",
    "Intervention",
]

DATASET_CANDIDATES = ["dataset.csv", "psychological_regulation_dataset.csv"]
MODEL_PATH = Path("model.pkl")


def _resolve_dataset_path() -> Path:
    for candidate in DATASET_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path
    return Path(DATASET_CANDIDATES[0])


def _get_feature_columns() -> List[str]:
    return [col for cols in FEATURES.values() for col in cols]


def _split_features(
    data: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    categorical_cols = [col for col in CATEGORICAL_FEATURES if col in feature_cols]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]
    return data[feature_cols], data[TARGETS], numeric_cols, categorical_cols


def _normalize_intervention_response(data: pd.DataFrame) -> pd.DataFrame:
    if "Intervention_Response" not in data.columns:
        return data

    series = data["Intervention_Response"]
    numeric_series = pd.to_numeric(series, errors="coerce")
    numeric_ratio = numeric_series.notna().mean()

    if pd.api.types.is_numeric_dtype(series) or numeric_ratio > 0.8:
        data = data.copy()
        data["Intervention_Response"] = pd.cut(
            numeric_series,
            bins=[-float("inf"), 0.33, 0.66, float("inf")],
            labels=["Negative", "Neutral", "Positive"],
        ).astype(str)
    return data


def _build_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    classifier = DecisionTreeClassifier(max_depth=10, min_samples_split=20)
    model = MultiOutputClassifier(classifier)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def _log_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> None:
    accuracies = []
    for idx, target in enumerate(TARGETS):
        target_accuracy = accuracy_score(y_true.iloc[:, idx], y_pred.iloc[:, idx])
        accuracies.append(target_accuracy)
        LOGGER.info("Accuracy for %s: %.4f", target, target_accuracy)

    if accuracies:
        LOGGER.info("Mean accuracy: %.4f", sum(accuracies) / len(accuracies))
    for idx, target in enumerate(TARGETS):
        report = classification_report(
            y_true.iloc[:, idx], y_pred.iloc[:, idx], zero_division=0
        )
        LOGGER.info("Classification report for %s:\n%s", target, report)


def _log_top_rules(pipeline: Pipeline, feature_cols: List[str]) -> None:
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = list(preprocessor.get_feature_names_out(feature_cols))
    estimators = pipeline.named_steps["model"].estimators_
    for target, estimator in zip(TARGETS, estimators):
        rules_text = export_text(estimator, feature_names=feature_names)
        rule_lines = [line for line in rules_text.splitlines() if "class" in line]
        top_rules = rule_lines[:5]
        LOGGER.info("Top rules for %s:", target)
        for rule in top_rules:
            LOGGER.info("%s", rule)


def train() -> None:
    dataset_path = _resolve_dataset_path()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found. Expected one of: {DATASET_CANDIDATES}"
        )

    data = pd.read_csv(dataset_path)
    if "Intervention" not in data.columns and "Previous_Intervention" in data.columns:
        LOGGER.warning(
            "Dataset is missing 'Intervention'; using 'Previous_Intervention' instead."
        )
        data = data.rename(columns={"Previous_Intervention": "Intervention"})
    feature_cols = _get_feature_columns()
    data = _normalize_intervention_response(data)
    missing_cols = [col for col in feature_cols + TARGETS if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    x, y, numeric_cols, categorical_cols = _split_features(data, feature_cols)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y[TARGETS[0]]
    )

    pipeline = _build_pipeline(numeric_cols, categorical_cols)
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_pred_df = pd.DataFrame(y_pred, columns=TARGETS)
    _log_metrics(y_test, y_pred_df)
    _log_top_rules(pipeline, feature_cols)

    joblib.dump(pipeline, MODEL_PATH)
    LOGGER.info("Model saved to %s", MODEL_PATH)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


if __name__ == "__main__":
    _configure_logging()
    train()
