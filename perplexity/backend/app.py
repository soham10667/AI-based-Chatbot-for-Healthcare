from flask import Flask, request, jsonify
from flask_cors import CORS

# ---------- IMPORT YOUR BACKEND LOGIC ----------

# correct names from predict_disease.py
from predict_disease import (
    load_data_and_models,
    predict_disease_with_details,
)

# from medicine_io3.py (these names must match exactly what is defined there)
from medicine_io3 import (
    recommend_medicines_from_symptoms,
    build_chatbot_answer,
)

app = Flask(__name__)
CORS(app)

# ---------- LOAD MODELS/DATA ONCE ----------

# uses load_data_and_models from predict_disease.py
disease_model, symptoms_list, disease_desc_dict, precaution_dict, severity_dict = (
    load_data_and_models()
)


@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "Healthcare chatbot backend running"}), 200


# ---------- 1. DISEASE PREDICTION API ----------

@app.route("/api/predict_disease", methods=["POST"])
def api_predict_disease():
    """
    Input JSON:
    {
      "symptoms": "itching, skin_rash, fever"
      // or
      "symptoms": ["itching", "skin_rash", "fever"]
    }
    """
    data = request.get_json(silent=True) or {}

    symptoms = data.get("symptoms")
    if not symptoms:
        return jsonify({"error": "symptoms field is required"}), 400

    # normalize to list of strings
    if isinstance(symptoms, str):
        inputsymptoms = [s.strip() for s in symptoms.split(",") if s.strip()]
    elif isinstance(symptoms, list):
        inputsymptoms = [str(s).strip() for s in symptoms if str(s).strip()]
    else:
        return jsonify({"error": "symptoms must be string or list"}), 400

    if not inputsymptoms:
        return jsonify({"error": "no valid symptoms provided"}), 400

    # correct function name: predict_disease_with_details(...)
    prediction, confidence, disease_desc, disease_precautions, symptom_severity = (
        predict_disease_with_details(
            disease_model,
            symptoms_list,
            disease_desc_dict,
            precaution_dict,
            severity_dict,
            inputsymptoms,
        )
    )

    # in predict_disease.py, symptom_severity is a list of "symptom: severity" strings
    severity_list = []
    for item in symptom_severity:
        # split "symptom: severity"
        if ":" in item:
            symptom, sev = item.split(":", 1)
            severity_list.append(
                {
                    "symptom": symptom.strip(),
                    "severity": sev.strip(),
                }
            )
        else:
            severity_list.append({"symptom": item, "severity": ""})

    return jsonify(
        {
            "success": True,
            "prediction": prediction,
            "confidence": confidence,
            "description": disease_desc,
            "precautions": disease_precautions,
            "symptom_severity": severity_list,
        }
    ), 200


# ---------- 2. MEDICINE RECOMMENDATION API ----------

@app.route("/api/recommend_medicines", methods=["POST"])
def api_recommend_medicines():
    """
    Input JSON:
    {
      "symptoms": "fever, headache",
      "topk": 5   // optional
    }
    """
    data = request.get_json(silent=True) or {}

    symptoms_text = data.get("symptoms", "")
    topk = int(data.get("topk", 3))

    if not symptoms_text.strip():
        return jsonify({"error": "symptoms field is required"}), 400

    try:
        medicines = recommend_medicines_from_symptoms(symptoms_text, topk)
    except Exception as e:
        return jsonify({"error": f"internal error: {e}"}), 500

    return jsonify(
        {
            "success": True,
            "symptoms": symptoms_text,
            "topk": topk,
            "medicines": medicines,
        }
    ), 200


# ---------- 3. CHATBOT TEXT ANSWER API ----------

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Input JSON:
    {
      "symptoms": "fever, headache, body pain"
    }
    """
    data = request.get_json(silent=True) or {}

    symptoms_text = data.get("symptoms", "")
    if not symptoms_text.strip():
        return jsonify({"error": "symptoms field is required"}), 400

    try:
        message = build_chatbot_answer(symptoms_text, 3)
    except Exception as e:
        return jsonify({"error": f"internal error: {e}"}), 500

    return jsonify(
        {
            "success": True,
            "symptoms": symptoms_text,
            "message": message,
        }
    ), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
