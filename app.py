import os
import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='static', static_url_path='')

# Load the trained model globally for fast predictions
MODEL_PATH = os.path.join("models", "sales_model.pkl")
METRICS_PATH = os.path.join("models", "metrics.json")
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Model file not found at {MODEL_PATH}. Prediction service will be offline.")

# Load model initially
load_model()

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        # Attempt to reload if it wasn't available initially
        load_model()
        if model is None:
            return jsonify({"error": "Prediction model is not available. Please run train.py first."}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Extract features
        tv = float(data.get('tv', 0))
        radio = float(data.get('radio', 0))
        newspaper = float(data.get('newspaper', 0))

        # Check ranges
        if tv < 0 or radio < 0 or newspaper < 0:
            return jsonify({"warning": "Budgets should be non-negative values", "tv": tv, "radio": radio, "newspaper": newspaper}), 400

        # Perform prediction
        input_df = pd.DataFrame([{
            'TV': tv,
            'Radio': radio,
            'Newspaper': newspaper
        }])
        
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            "success": True,
            "prediction": round(prediction, 4),
            "input": {
                "tv": tv,
                "radio": radio,
                "newspaper": newspaper
            }
        })

    except ValueError as ve:
        return jsonify({"error": f"Invalid data format: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    if not os.path.exists(METRICS_PATH):
        return jsonify({"error": "Metrics file not found. Please train the models."}), 404
        
    try:
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": f"Failed to load metrics: {str(e)}"}), 500

if __name__ == '__main__':
    # Running Flask app on port 8000
    app.run(debug=True, host='127.0.0.1', port=8000)
