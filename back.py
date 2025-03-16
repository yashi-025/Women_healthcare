import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler

# Load trained model & scaler
with open("healthcare_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define AI-based health suggestions
health_suggestions = {
    0: "Maintain a healthy diet and exercise regularly.",
    1: "Increase iron-rich foods, take supplements if needed, and monitor hemoglobin levels.",
    2: "Manage stress, get enough sleep, and seek therapy if necessary.",
    3: "Follow a balanced diet and engage in daily physical activity to prevent obesity.",
}

# Create Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input
        data = request.json
        user_features = data["features"]

        # Convert input to NumPy array and scale
        user_features_scaled = scaler.transform([user_features])

        # Make prediction
        prediction = model.predict(user_features_scaled)[0]

        # Get simple health advice
        advice = health_suggestions.get(prediction, "Please consult a doctor for further guidance.")

        # Return response
        return jsonify({
            "predicted_disease_cluster": int(prediction),
            "health_advice": advice
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
