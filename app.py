import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

app = Flask(__name__)
CORS(app)

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
print("🚀 Booting up Energy AI...")

MODEL_FILE = 'trained_model.pkl'
SCALER_FILE = 'scaler.pkl'
ENCODER_FILE = 'encoders.pkl'
csv_path = 'sigma.csv'

# Initialize global-level placeholders
le_state = LabelEncoder()
le_sector = LabelEncoder()
scaler = StandardScaler()
model_std_dev = 0.4 

try:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")
    df = pd.read_csv(csv_path)
    print("✅ CSV data loaded.")
except FileNotFoundError:
    print("⚠️ CSV missing! Creating dummy data...")
    data = {
        'year': [2020]*10, 'month': [1,2,3,4,5,6,7,8,9,10],
        'stateDescription': ['alabama']*10, 'sectorName': ['residential']*10,
        'price': [12.5, 12.7, 12.6, 12.9, 13.0, 13.1, 13.2, 13.5, 13.3, 13.6]
    }
    df = pd.DataFrame(data)

# Normalize strings immediately
df['stateDescription'] = df['stateDescription'].astype(str).str.strip().str.lower()
df['sectorName'] = df['sectorName'].astype(str).str.strip().str.lower()

# ==========================================
# 2. FAST MODEL TRAINING (Optimized for Render)
# ==========================================
if os.path.exists(MODEL_FILE):
    print("📂 Loading saved model...")
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    saved_assets = joblib.load(ENCODER_FILE)
    le_state = saved_assets['le_state']
    le_sector = saved_assets['le_sector']
    model_std_dev = saved_assets.get('std_dev', 0.4)
else:
    print("🧠 No model found. Training LIGHTWEIGHT version...")
    
    # Fast Encoding
    df['state_encoded'] = le_state.fit_transform(df['stateDescription'])
    df['sector_encoded'] = le_sector.fit_transform(df['sectorName'])
    
    X = df[['year', 'month', 'state_encoded', 'sector_encoded']]
    y = df['price']
    X_scaled = scaler.fit_transform(X)

    # LIGHTER Neural Network for 60-second boot limit
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32), # Reduced from 100,50,25
        activation='relu', 
        solver='adam', 
        max_iter=500,        # Reduced from 1500 to boot faster
        early_stopping=True, # Stops training when accuracy plateaus
        random_state=42
    )

    model.fit(X_scaled, y)
    
    # Standard deviation for variation logic
    preds = model.predict(X_scaled)
    model_std_dev = np.sqrt(mean_squared_error(y, preds))

    # Save so next boot is instant
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump({
        'le_state': le_state, 
        'le_sector': le_sector, 
        'std_dev': model_std_dev
    }, ENCODER_FILE)
    
    print("✅ Training complete. Port opening now...")

# ==========================================
# 3. API ROUTES
# ==========================================
@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "online", "engine": "Deep Learning (MLP)"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_state = data.get('state', '').strip().lower()
        input_sector = data.get('sector', '').strip().lower()
        start_year = int(data.get('year'))
        start_month = int(data.get('month'))

        if input_state not in le_state.classes_ or input_sector not in le_sector.classes_:
            return jsonify({'error': f'Unknown Location/Sector: {input_state}'}), 400

        state_code = le_state.transform([input_state])[0]
        sector_code = le_sector.transform([input_sector])[0]

        predictions = []
        cy, cm = start_year, start_month

        for _ in range(12):
            raw_input = np.array([[cy, cm, state_code, sector_code]])
            scaled_input = scaler.transform(raw_input)
            
            base_pred = model.predict(scaled_input)[0]
            # Use the model's error margin for realistic variation
            variation = np.random.normal(0, model_std_dev * 0.3)
            final_price = max(0.01, base_pred + variation)

            predictions.append({
                'month': cm, 'year': cy, 'price': round(float(final_price), 2)
            })

            cm += 1
            if cm > 12:
                cm = 1
                cy += 1

        return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
