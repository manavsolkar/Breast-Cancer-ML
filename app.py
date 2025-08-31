# src/app.py
from flask import Flask, request, jsonify, render_template_string
import joblib
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE / "models"
MODEL_PATH = MODEL_DIR / "best_model.joblib"

app = Flask(__name__)

# Simple HTML form to input 30 features
HTML = """
<!doctype html>
<title>Breast Cancer Inference</title>
<h2>Breast Cancer Prediction (Malignant=0, Benign=1)</h2>
<form method=post>
  <label>Comma-separated 30 feature values:</label><br>
  <textarea name=row rows=4 cols=80>{{ example }}</textarea><br><br>
  <input type=submit value="Predict">
</form>
{% if result %}
  <h3>Result</h3>
  <p>Predicted class: <b>{{ result.pred }}</b></p>
  <p>Probability (benign): <b>{{ result.proba }}</b></p>
{% endif %}
"""

model_bundle = None

def load_model():
    global model_bundle
    if model_bundle is None:
        model_bundle = joblib.load(MODEL_PATH)
    return model_bundle

@app.route("/", methods=["GET", "POST"])
def home():
    example = "14.5,20.1,94.5,600.1,0.102,0.11,0.08,0.02,0.18,0.06,0.25,0.75,1.7,23.0,0.006,0.02,0.02,0.004,0.016,0.003,16.0,25.0,110.0,800.0,0.14,0.3,0.25,0.07,0.28,0.08"
    result = None
    if request.method == "POST":
        row = request.form.get("row", "")
        try:
            arr = np.array([float(x) for x in row.split(",")], dtype=float).reshape(1,-1)
            bundle = load_model()
            model = bundle["model"]
            proba = None
            try:
                proba = model.predict_proba(arr)[0,1]
            except Exception:
                try:
                    proba = model.decision_function(arr)[0]
                except Exception:
                    proba = None
            pred = int(proba >= 0.5) if proba is not None else int(model.predict(arr)[0])
            result = {"pred": pred, "proba": float(proba) if proba is not None else None}
        except Exception as e:
            result = {"error": str(e)}
    return render_template_string(HTML, example=example, result=result)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    if "row" not in data:
        return jsonify({"error": "Please provide 'row' as a list or comma-separated string."}), 400
    row = data["row"]
    if isinstance(row, str):
        row = [float(x) for x in row.split(",")]
    arr = np.array(row, dtype=float).reshape(1,-1)
    bundle = load_model()
    model = bundle["model"]
    proba = None
    try:
        proba = model.predict_proba(arr)[0,1]
    except Exception:
        try:
            proba = model.decision_function(arr)[0]
        except Exception:
            proba = None
    pred = int(proba >= 0.5) if proba is not None else int(model.predict(arr)[0])
    return jsonify({"pred": pred, "proba": proba})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
