import joblib
import pickle
import os

model_path = 'models/final_model.pkl'
scaler_path = 'models/scaler.pkl'

print(f"Original Model Size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

# Load existing model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Save with compression
joblib.dump(model, model_path, compress=3)

print(f"Compressed Model Size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

# Re-save scaler with joblib just for consistency (optional but good)
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
joblib.dump(scaler, scaler_path, compress=3)
