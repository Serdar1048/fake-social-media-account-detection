import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report

def main():
    # Load Model and Data
    try:
        model = joblib.load('models/final_model.pkl')
        X_test = pd.read_csv('datasets/processed/X_test.csv')
        y_test = pd.read_csv('datasets/processed/y_test.csv')
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n--- Model Parameters ---")
    print(model.get_params())
    
    print("\n--- Functionality Metrics ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
