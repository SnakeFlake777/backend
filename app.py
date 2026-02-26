import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

app = Flask(__name__)
# Enable CORS so your CodeHS HTML frontend can talk to this server securely
CORS(app)

# ==========================================
# 1. LOAD DATA & PREPARE ENCODERS
# ==========================================
print("Initializing API and loading data...")

# Try to load the CSV. In a hosted environment, the CSV should be in the same folder.
csv_path = 'sigma.csv'

try:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")
    df = pd.read_csv(csv_path)
    print("✅ Data loaded successfully from CSV.")
except FileNotFoundError:
    print("⚠️ WARNING: CSV not found. Using fallback synthetic data.")
    # Fallback data to ensure the server doesn't crash if the file is missing
    data = {
        'year': [2020]*5 + [2021]*5,
        'month': [1, 2, 3, 4, 5]*2,
        'stateDescription': ['Alabama']*10,
        'sectorName': ['residential']*10,
        'price': [12.5, 12.7, 12.6, 12.9, 13.0, 13.1, 13.2, 13.5, 13.3, 13.6]
    }
    df = pd.DataFrame(data)

# Initialize Encoders and Scaler
le_state = LabelEncoder()
le_sector = LabelEncoder()
scaler = StandardScaler()

# Clean and encode the categorical data
df['state_encoded'] = le_state.fit_transform(df['stateDescription'].astype(str).str.strip().str.lower())
df['sector_encoded'] = le_sector.fit_transform(df['sectorName'].astype(str).str.strip().str.lower())

# Define Features (X) and Target (y)
X = df[['year', 'month', 'state_encoded', 'sector_encoded']]
y = df['price']

# Scale the inputs (Absolutely required for Neural Networks)
X_scaled = scaler.fit_transform(X)

# ==========================================
# 2. TRAIN THE DEEP LEARNING MODEL
# ==========================================
print("🧠 Training Deep Neural Network (MLP)...")
# 3 hidden layers with 100, 50, and 25 neurons
model = MLPRegressor(
    hidden_layer_sizes=(100, 50, 25), 
    activation='relu', 
    solver='adam', 
    max_iter=1500, 
    random_state=42
)
model.fit(X_scaled, y)
print("✅ Model trained successfully!")

# Calculate base variance/error to use for our realistic prediction fluctuations
predictions_on_train = model.predict(X_scaled)
mse = mean_squared_error(y, predictions_on_train)
model_std_dev = np.sqrt(mse) 

# ==========================================
# 3. DEFINE API ROUTES
# ==========================================
@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "online", "message": "Deep Learning Pricing API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # 1. Parse and clean incoming user data
        input_state = data.get('state', '').strip().lower()
        input_sector = data.get('sector', '').strip().lower()
        start_year = int(data.get('year'))
        start_month = int(data.get('month'))

        # 2. Validate against known classes
        if input_state not in le_state.classes_ or input_sector not in le_sector.classes_:
            return jsonify({
                'error': f'Invalid State or Sector: "{input_state}", "{input_sector}".'
            }), 400

        # 3. Encode the validated inputs
        state_code = le_state.transform([input_state])[0]
        sector_code = le_sector.transform([input_sector])[0]

        predictions = []
        current_year = start_year
        current_month = start_month

        # 4. Generate 12 months of predictions
        for _ in range(12):
            # Create raw array for this specific month
            raw_input_data = np.array([[current_year, current_month, state_code, sector_code]])
            
            # Scale the input using the scaler fitted during training
            scaled_input = scaler.transform(raw_input_data)
            
            # Get the Neural Network's baseline prediction
            base_pred = model.predict(scaled_input)[0]
            
            # Add mathematical variance (fluctuation) based on the model's historical error margin
            # (Scale of 0.3 dampens the wildness of the randomness)
            varied_pred = np.random.normal(loc=base_pred, scale=model_std_dev * 0.3)
            
            # Ensure price never drops below a logical threshold
            final_price = max(0.01, varied_pred)

            predictions.append({
                'month': current_month,
                'year': current_year,
                'price': round(final_price, 2)
            })

            # Advance the calendar
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1

        return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==========================================
# 4. SERVER EXECUTION (For local testing)
# ==========================================
# In production (Render), a WSGI server like Gunicorn will run the app, 
# so this block is mostly for testing on your local machine.
if __name__ == '__main__':
    # Use the PORT environment variable provided by the host, default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
