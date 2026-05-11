"""Flask API server for DSS stress predictions."""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from utils.predictor import Predictor

LOGGER = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

PREDICTOR = Predictor(model_path="model.pkl")

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


if __name__ == "__main__":
    _configure_logging()
    app.run(host="0.0.0.0", port=5000, debug=True)
