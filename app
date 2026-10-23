import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")

app = Flask(__name__)
CORS(app, resources={
    r"/predict": {"origins": ["https://edupredictmonitoring.netlify.app"]}
})

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("model.joblib is missing. Train the model in Google Colab and add model.joblib beside app.py.")

model = joblib.load(MODEL_PATH)
FEATURE_COLUMNS = ["age", "sex", "address", "studytime", "failures", "activities", "absences", "G1", "G2"]

@app.get("/")
def home():
    return jsonify({"status": "ok", "service": "EduPredict ML API", "model": "RandomForestRegressor"})

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    try:
        row = {
            "age": int(data["age"]),
            "sex": str(data["sex"]).upper(),
            "address": str(data["address"]).upper(),
            "studytime": int(data["studytime"]),
            "failures": int(data["failures"]),
            "activities": str(data["activities"]).lower(),
            "absences": int(data["absences"]),
            "G1": float(data["G1"]),
            "G2": float(data["G2"]),
        }
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Missing or invalid input. Please check all student fields."}), 400

    if not 15 <= row["age"] <= 22:
        return jsonify({"error": "Age must be between 15 and 22 for this dataset."}), 400
    if row["sex"] not in {"F", "M"}:
        return jsonify({"error": "Sex must be F or M for the selected dataset."}), 400
    if row["address"] not in {"U", "R"}:
        return jsonify({"error": "Address must be U (urban) or R (rural)."}), 400
    if not 1 <= row["studytime"] <= 4:
        return jsonify({"error": "Study time must be from 1 to 4."}), 400
    if not 0 <= row["failures"] <= 3:
        return jsonify({"error": "Past failures must be from 0 to 3."}), 400
    if row["activities"] not in {"yes", "no"}:
        return jsonify({"error": "Activities must be yes or no."}), 400
    if not 0 <= row["absences"] <= 93:
        return jsonify({"error": "Absences must be from 0 to 93."}), 400
    if not 0 <= row["G1"] <= 20 or not 0 <= row["G2"] <= 20:
        return jsonify({"error": "Series grades must be from 0 to 20."}), 400

    features = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    grade = float(model.predict(features)[0])
    grade = max(0.0, min(20.0, grade))
    percent = grade / 20.0 * 100.0

    # These are project-defined display bands, not official UCI categories.
    if grade >= 14:
        label, risk = "High Performance", "Low Risk"
    elif grade >= 10:
        label, risk = "Moderate Performance", "Medium Risk"
    else:
        label, risk = "Low Performance", "High Risk"

    return jsonify({
        "predicted_grade": round(grade, 2),
        "score_percent": round(percent, 1),
        "label": label,
        "risk": risk,
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
