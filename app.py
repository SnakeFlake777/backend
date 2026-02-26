import os
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the brain immediately
try:
    model = joblib.load('trained_model.pkl')
    scaler = joblib.load('scaler.pkl')
    assets = joblib.load('encoders.pkl')
    le_state = assets['le_state']
    le_sector = assets['le_sector']
    model_std_dev = assets['std_dev']
    print("🚀 Model loaded. System online.")
except Exception as e:
    print(f"❌ ERROR: Could not load model files: {e}")

@app.route('/')
def home():
    return jsonify({"status": "live"})

@app.route('/predict', methods=['POST'])
def predict():
    # ... (Keep your existing /predict logic here) ...
    pass

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
