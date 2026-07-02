from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'Churn.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, 'standard_scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Build feature vector in correct order
        # HTML sends TotalCharges_rep_yeo_CapMS but model expects TotalChargeshmv_yeo_CapMS
        feature_map = {
            'TotalChargeshmv_yeo_CapMS': data.get('TotalCharges_rep_yeo_CapMS', data.get('TotalChargeshmv_yeo_CapMS', 0))
        }

        features = []
        for feat in FEATURE_ORDER:
            if feat in feature_map:
                features.append(float(feature_map[feat]))
            else:
                features.append(float(data.get(feat, 0)))

        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]

        churn_prob = round(float(proba[1]) * 100, 2)
        retention_prob = round(float(proba[0]) * 100, 2)

        return jsonify({
            'prediction': 'Will Churn' if prediction == 1 else 'Will Not Churn',
            'churn_probability': churn_prob,
            'retention_probability': retention_prob,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '__main__':
    app.run(debug=True)