import pandas as pd
import joblib
from flask import Flask, request, jsonify
import numpy as np

# Initialize Flask App
app = Flask(__name__)

# Model File Path
MODEL_PATH = "trained-model.sav"

# Load Models & Scaler
def load_models():
    try:
        data = joblib.load(MODEL_PATH)
        models = data["models"]
        scaler = data["scaler"]
        print("✅ Models and Scaler loaded successfully.")
        return models, scaler
    except Exception as e:
        print(f"❌ Failed to load models or scaler: {e}")
        exit()

# Load the models and scaler
models, scaler = load_models()

# Expected Feature Columns
FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Validate Input Data
def validate_input(data):
    for feature in FEATURES:
        if feature not in data or data[feature] is None or data[feature] == '':
            return f"Missing or invalid value for field: {feature}"
    if data['Pregnancies'] < 0 or data['Age'] < 0:
        return "Pregnancies and Age should not be negative."
    if data['Glucose'] <= 0 or data['BMI'] <= 0 or data['BloodPressure'] <= 0:
        return "Glucose, BMI, and BloodPressure should be positive values."
    return None

# Determine Risk Level and Recommendation
def determine_risk(probability):
    if probability >= 70:
        return "High", "Consult a doctor immediately and monitor your blood sugar levels."
    elif probability >= 40:
        return "Medium", "Consider lifestyle changes like a balanced diet and regular exercise."
    return "Low", "Maintain a healthy lifestyle."

# Prediction API Route
@app.route('/predict', methods=['POST'])
def predict_diabetes():
    try:
        print("📨 Received a new prediction request...")
        data = request.get_json()
        print("📥 Raw input data:", data)

        # Validate input data
        validation_error = validate_input(data)
        if validation_error:
            print(f"❌ {validation_error}")
            return jsonify({"error": validation_error}), 400

        # Convert input to DataFrame
        user_data = pd.DataFrame([data])[FEATURES]

        # Preprocess the data
        user_data_scaled = scaler.transform(user_data)

        # Collect predictions from all models
        predictions = {}
        for name, model in models.items():
            if name in ["SVM", "KNN", "Logistic Regression"]:
                prediction = model.predict(user_data_scaled)
                precentage = round(model.predict_proba(user_data_scaled)[0][1] * 100, 2)
            else:
                prediction = model.predict(user_data)
                precentage = round(model.predict_proba(user_data)[0][1] * 100, 2)

            risk_level, recommendation = determine_risk(precentage)
            predictions[name] = {
                "prediction": bool(prediction[0]),
                "precentage": precentage,  # Changed from 'percentage' to 'precentage'
                "riskLevel": risk_level,
                "recommendation": recommendation
            }

        print("📊 Predictions:", predictions)
        return jsonify(predictions)

    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 500

# Run Flask Server
if __name__ == '__main__':
    print("🚀 Starting Flask server on port 3002...")
    app.run(host='0.0.0.0', port=3002, debug=True)
