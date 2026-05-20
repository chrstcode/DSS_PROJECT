"""Flask API server for DSS stress predictions."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from flask_cors import CORS

import pandas as pd

from utils.predictor import Predictor

LOGGER = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR = Predictor(model_path=os.path.join(BASE_DIR, "model.pkl"))

DATASET_CANDIDATES = ["dataset.csv", "psychological_regulation_dataset.csv"]
DATASET_PREVIEW_ROWS = 20

REQUIRED_FIELDS = {
    "stress_score": (0, 100),
    "anxiety_score": (0, 100),
    "exam_pressure": (0, 10),
    "sleep_hours": (0, 24),
    "social_support": (0, 10),
    "heart_rate": (40, 200),
    "physical_activity": (0, 10),
    "assignment_load": (0, 10),
    "study_hours": (0, 24),
    "attendance": (0, 100),
    "screen_time": (0, 24),
    "facial_emotion": None,
    "mood_state": None,
    "intervention_response": None,
    "reward_score": (0, 100),
}


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _resolve_dataset_path() -> Path:
    for candidate in DATASET_CANDIDATES:
        path = Path(BASE_DIR) / candidate
        if path.exists():
            return path
    return Path(BASE_DIR) / DATASET_CANDIDATES[0]


@lru_cache(maxsize=1)
def _load_dataset_preview() -> Tuple[List[str], List[Dict[str, Any]], int]:
    dataset_path = _resolve_dataset_path()
    if not dataset_path.exists():
        return [], [], 0

    data = pd.read_csv(dataset_path)
    preview = data.head(DATASET_PREVIEW_ROWS).copy()
    preview = preview.where(pd.notna(preview), None)
    columns = list(preview.columns)
    rows = preview.to_dict(orient="records")
    return columns, rows, int(len(data))


def _validate_payload(payload: Dict[str, Any]) -> Tuple[bool, str]:
    for field in REQUIRED_FIELDS:
        if field not in payload:
            return False, f"Missing required field: {field}"

    for field, limits in REQUIRED_FIELDS.items():
        value = payload.get(field)
        if limits is None:
            if not isinstance(value, str) or not value.strip():
                return False, f"{field} must be a non-empty string"
            continue

        if not isinstance(value, (int, float)):
            return False, f"{field} must be a number"

        min_val, max_val = limits
        if value < min_val or value > max_val:
            return False, f"{field} must be between {min_val} and {max_val}"

    return True, ""


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify({"status": "healthy", "model_loaded": PREDICTOR.is_ready})


@app.route("/", methods=["GET"])
def index() -> Any:
    return render_template("index.html")


@app.route("/logo.png")
def logo_image() -> Any:
    """Serve the project logo.png from the project root for templates.

    The file `logo.png` is stored at the repository root. Exposing it
    here allows templates to reference it using `url_for('logo_image')`.
    """
    try:
        return send_from_directory(BASE_DIR, "logo.png")
    except Exception:
        return ("", 404)


@app.route("/predict", methods=["POST"])
def predict() -> Any:
    try:
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            return jsonify({"success": False, "error": "Invalid JSON payload"}), 400

        is_valid, message = _validate_payload(payload)
        if not is_valid:
            return jsonify({"success": False, "error": message}), 400

        result = PREDICTOR.predict(payload)
        return jsonify(result)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Prediction failed")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/tree", methods=["GET"])
def tree() -> Any:
    if not PREDICTOR.is_ready:
        return jsonify({"success": False, "error": "Model is not loaded"}), 500
    trees = PREDICTOR.get_tree_dots()
    return jsonify({"success": True, "trees": trees})


@app.route("/dataset-preview", methods=["GET"])
def dataset_preview() -> Any:
    columns, rows, total_rows = _load_dataset_preview()
    if not columns:
        return jsonify({"success": False, "error": "Dataset not found"}), 404

    return jsonify(
        {
            "success": True,
            "columns": columns,
            "rows": rows,
            "total_rows": total_rows,
        }
    )


if __name__ == "__main__":
    _configure_logging()
    app.run(host="0.0.0.0", port=5000, debug=True)
