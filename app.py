import numpy as np
import pandas as pd
import pickle
import os
import sys
from flask import Flask, render_template, request

import logging
from logging_code import setup_logging
logger = setup_logging('app')

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# loading the saved model
try:
    with open('Churn.pkl', 'rb') as f:
        model = pickle.load(f)
    logger.info(f'Model loaded successfully : {model}')
except Exception as e:
    logger.info(f'Error loading model : {e}')
    model = None


@app.route('/', methods=['GET'])
def home():
    """Renders the home page with the prediction form."""
    try:
        logger.info(f'Home page loaded')
        return render_template('index.html', prediction=None)
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives form data, prepares input features and returns churn prediction.
    """
    try:
        logger.info(f'=================== Prediction Request Received ===================')

        # ---- collecting form inputs ----
        # numerical
        SeniorCitizen   = int(request.form['SeniorCitizen'])
        tenure          = float(request.form['tenure'])
        MonthlyCharges  = float(request.form['MonthlyCharges'])
        TotalCharges    = float(request.form['TotalCharges'])

        # categorical - OHE (gender, Partner, Dependents)
        gender      = request.form['gender']          # Male / Female
        Partner     = request.form['Partner']          # Yes / No
        Dependents  = request.form['Dependents']       # Yes / No

        # categorical - Ordinal
        PhoneService    = request.form['PhoneService']
        MultipleLines   = request.form['MultipleLines']
        InternetService = request.form['InternetService']
        OnlineSecurity  = request.form['OnlineSecurity']
        OnlineBackup    = request.form['OnlineBackup']
        DeviceProtection= request.form['DeviceProtection']
        TechSupport     = request.form['TechSupport']
        StreamingTV     = request.form['StreamingTV']
        StreamingMovies = request.form['StreamingMovies']
        Contract        = request.form['Contract']
        PaperlessBilling= request.form['PaperlessBilling']
        PaymentMethod   = request.form['PaymentMethod']
        Sim             = request.form['Sim']

        logger.info(f'Form inputs collected successfully')

        # ---- OHE manually (drop=first so Male=0, Female=1 | No=0, Yes=1) ----
        gender_Male      = 1 if gender == 'Male' else 0
        Partner_Yes      = 1 if Partner == 'Yes' else 0
        Dependents_Yes   = 1 if Dependents == 'Yes' else 0

        # ---- Ordinal encoding manually (alphabetical order - sklearn default) ----
        phone_map    = {'No': 0, 'Yes': 1}
        lines_map    = {'No': 0, 'No phone service': 1, 'Yes': 2}
        internet_map = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
        yesno_map    = {'No': 0, 'No internet service': 1, 'Yes': 2}
        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        payment_map  = {'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1, 'Electronic check': 2, 'Mailed check': 3}
        sim_map      = {'Airtel': 0, 'BSNL': 1, 'Jio': 2, 'Vi': 3}
        paper_map    = {'No': 0, 'Yes': 1}

        PhoneService_ord     = phone_map[PhoneService]
        MultipleLines_ord    = lines_map[MultipleLines]
        InternetService_ord  = internet_map[InternetService]
        OnlineSecurity_ord   = yesno_map[OnlineSecurity]
        OnlineBackup_ord     = yesno_map[OnlineBackup]
        DeviceProtection_ord = yesno_map[DeviceProtection]
        TechSupport_ord      = yesno_map[TechSupport]
        StreamingTV_ord      = yesno_map[StreamingTV]
        StreamingMovies_ord  = yesno_map[StreamingMovies]
        Contract_ord         = contract_map[Contract]
        PaperlessBilling_ord = paper_map[PaperlessBilling]
        PaymentMethod_ord    = payment_map[PaymentMethod]
        Sim_ord              = sim_map[Sim]

        logger.info(f'Encoding done successfully')

        # ---- building input array in same order as training ----
        input_data = [[
            SeniorCitizen, tenure, MonthlyCharges, TotalCharges,
            gender_Male, Partner_Yes, Dependents_Yes,
            PhoneService_ord, MultipleLines_ord, InternetService_ord,
            OnlineSecurity_ord, OnlineBackup_ord, DeviceProtection_ord,
            TechSupport_ord, StreamingTV_ord, StreamingMovies_ord,
            Contract_ord, PaperlessBilling_ord, PaymentMethod_ord, Sim_ord
        ]]

        logger.info(f'Input data prepared : {input_data}')

        # ---- prediction ----
        prediction = model.predict(input_data)[0]
        logger.info(f'Raw prediction : {prediction}')

        # 0 = Yes Churn, 1 = No Churn (as per your y mapping in main.py)
        result = 'Will NOT Churn' if prediction == 1 else 'Will Churn'
        logger.info(f'Final Result : {result}')

        return render_template('index.html', prediction=result)

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')
        return render_template('index.html', prediction='Error occurred. Please check inputs.')


if __name__ == '__main__':
    app.run(debug=True)
