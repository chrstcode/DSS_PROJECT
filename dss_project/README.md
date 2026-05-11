# DSS Project - Student Stress Prediction

## Overview
This project trains a Decision Tree model to predict stress-related outcomes for university students and provides a Flask API for real-time predictions.

## Project Structure
```
dss_project/
├── app.py
├── train_model.py
├── model.pkl
├── dataset.csv
├── requirements.txt
├── README.md
└── utils/
    ├── __init__.py
    └── predictor.py
```

## Setup
1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your dataset in the project root as `dataset.csv`.
   - If you have `psychological_regulation_dataset.csv`, you can keep the same name and the training script will detect it.

## Train the Model
```bash
python train_model.py
```
This will generate `model.pkl` in the project root and print training metrics plus top decision rules.

## Run the API Server
```bash
python app.py
```
The server runs at `http://localhost:5000`.

## Example Requests
### Health Check
```bash
curl http://localhost:5000/health
```

### Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "stress_score": 65,
    "anxiety_score": 58,
    "exam_pressure": 7,
    "sleep_hours": 5.5,
    "social_support": 6,
    "heart_rate": 82,
    "physical_activity": 3,
    "assignment_load": 8,
    "study_hours": 4,
    "attendance": 85,
    "screen_time": 6.5,
    "facial_emotion": "Neutral",
    "mood_state": "Anxious",
    "intervention_response": "Positive",
    "reward_score": 45
  }'
```

### Python requests
```python
import requests

payload = {
    "stress_score": 65,
    "anxiety_score": 58,
    "exam_pressure": 7,
    "sleep_hours": 5.5,
    "social_support": 6,
    "heart_rate": 82,
    "physical_activity": 3,
    "assignment_load": 8,
    "study_hours": 4,
    "attendance": 85,
    "screen_time": 6.5,
    "facial_emotion": "Neutral",
    "mood_state": "Anxious",
    "intervention_response": "Positive",
    "reward_score": 45,
}

response = requests.post("http://localhost:5000/predict", json=payload)
print(response.json())
```

## Notes
- The model uses a Decision Tree with `max_depth=10` and `min_samples_split=20`.
- Input validation enforces required fields and numeric ranges.
- Explanations are heuristic and meant for interpretability.
