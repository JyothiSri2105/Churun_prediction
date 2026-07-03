from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import traceback

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model
try:
    with open(os.path.join(BASE_DIR, "Churn.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(BASE_DIR, "standard_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    print("Model loaded successfully")

except Exception as e:
    print("Error loading model")
    print(e)
    model = None
    scaler = None


FEATURE_ORDER = [
    'SeniorCitizen',
    'tenure_yeo_tri',
    'MonthlyCharges_box_CapMS',
    'TotalChargeshmv_yeo_CapMS',
    'gender_Male',
    'Partner_Yes',
    'Dependents_Yes',
    'PhoneService_ordinal',
    'MultipleLines_ordinal',
    'InternetService_ordinal',
    'OnlineSecurity_ordinal',
    'OnlineBackup_ordinal',
    'DeviceProtection_ordinal',
    'TechSupport_ordinal',
    'StreamingTV_ordinal',
    'StreamingMovies_ordinal',
    'Contract_ordinal',
    'PaperlessBilling_ordinal',
    'PaymentMethod_ordinal',
    'Sim_ordinal'
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        if model is None:
            return jsonify({
                "status": "error",
                "error": "Model not loaded."
            }), 500

        data = request.get_json()

        feature_map = {
            "TotalChargeshmv_yeo_CapMS":
                data.get(
                    "TotalCharges_rep_yeo_CapMS",
                    data.get("TotalChargeshmv_yeo_CapMS", 0)
                )
        }

        features = []

        for feature in FEATURE_ORDER:

            if feature in feature_map:
                value = feature_map[feature]
            else:
                value = data.get(feature, 0)

            features.append(float(value))

        X = np.array(features).reshape(1, -1)

        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)[0]

        probability = model.predict_proba(X_scaled)[0]

        churn = round(probability[1] * 100, 2)
        retention = round(probability[0] * 100, 2)

        return jsonify({
            "status": "success",
            "prediction": "Will Churn" if prediction == 1 else "Will Not Churn",
            "churn_probability": churn,
            "retention_probability": retention
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.errorhandler(404)
def page_not_found(e):
    return "404 Page Not Found", 404


@app.errorhandler(500)
def internal_server_error(e):
    return "500 Internal Server Error", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)