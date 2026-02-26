import os
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global variables
model = None
scaler = None
le_state = None
le_sector = None
model_std_dev = 0.4
model_ready = False

def load_model():
    global model, scaler, le_state, le_sector, model_std_dev, model_ready
    try:
        # These filenames MUST match what you uploaded to GitHub
        model = joblib.load('trained_model.pkl')
        scaler = joblib.load('scaler.pkl')
        assets = joblib.load('encoders.pkl')
        
        le_state = assets['le_state']
        le_sector = assets['le_sector']
        model_std_dev = assets.get('std_dev', 0.4)
        
        print("🚀 Model loaded. System online.")
        model_ready = True
    except Exception as e:
        print(f"❌ ERROR: Could not load model files: {e}")
        model_ready = False

# Run the loader
load_model()

@app.route('/', methods=['GET', 'HEAD'])
def home():
    return jsonify({
        "status": "online" if model_ready else "error",
        "model_loaded": model_ready
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not model_ready:
        return jsonify({'error': 'Backend model is not loaded. Check server logs for NumPy version errors.'}), 500
        
    try:
        data = request.json
        input_state = data.get('state', '').strip().lower()
        input_sector = data.get('sector', '').strip().lower()
        start_year = int(data.get('year'))
        start_month = int(data.get('month'))

        # Validation
        if input_state not in le_state.classes_ or input_sector not in le_sector.classes_:
            return jsonify({'error': f'Invalid State/Sector: {input_state}'}), 400

        state_code = le_state.transform([input_state])[0]
        sector_code = le_sector.transform([input_sector])[0]

        predictions = []
        cy, cm = start_year, start_month

        for _ in range(12):
            raw_input = np.array([[cy, cm, state_code, sector_code]])
            scaled_input = scaler.transform(raw_input)
            
            # Predict
            base_pred = model.predict(scaled_input)[0]
            variation = np.random.normal(0, model_std_dev * 0.3)
            final_price = max(0.01, base_pred + variation)

            predictions.append({
                'month': cm,
                'year': cy,
                'price': round(float(final_price), 2)
            })

            cm += 1
            if cm > 12:
                cm = 1
                cy += 1

        return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
