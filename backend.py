import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS  # Add CORS support
import requests  # To fetch data from ESP32

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ESP32 Configuration
ESP32_SENSOR_URL = "http://192.168.4.1"  # Replace with your ESP32's IP address

# Load the dataset
file_path = "Crop_recommendation.csv"
df = pd.read_csv(file_path)

# Encode crop labels into numbers
encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["label"])

# Select only temperature, rainfall, ph, and humidity as features
X = df[["temperature", "rainfall", "ph", "humidity"]]
y = df["label"]  # Target variable

# Split the dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route('/fetch_sensor_data', methods=['GET'])
def fetch_sensor_data():
    try:
        # Fetch data from ESP32
        response = requests.get(ESP32_SENSOR_URL, timeout=5)
        
        if response.status_code == 200:
            sensor_data = response.json()
            return jsonify({
                "temperature": sensor_data["temperature"],
                "humidity": sensor_data["humidity"],
                "moisture": sensor_data["moisture"]
            })
        else:
            return jsonify({"error": "Could not fetch sensor data"}), 500
    
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("📥 Received Data:", data)

    try:
        # Convert input into DataFrame with correct column names
        input_data = pd.DataFrame([[
            float(data["temperature"]),
            float(data["rainfall"]),
            float(data["ph"]),
            float(data["humidity"])
        ]], columns=["temperature", "rainfall", "ph", "humidity"])

        print("🧐 Input for Model:", input_data)

        # Predict the crop
        predicted_label = model.predict(input_data)
        predicted_crop = encoder.inverse_transform(predicted_label)[0]

        # Calculate actual accuracy
        y_pred = model.predict(X_test)
        actual_accuracy = accuracy_score(y_test, y_pred) * 100

        print(f"🌱 Predicted Crop: {predicted_crop}, 🎯 Accuracy: {actual_accuracy:.2f}%")

        return jsonify({"crop": predicted_crop, "accuracy": f"{actual_accuracy:.2f}"})

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)