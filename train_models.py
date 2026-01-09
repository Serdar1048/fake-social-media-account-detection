import pandas as pd
import numpy as np
import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import xgboost as xgb

def load_data():
    print("Loading processed data...")
    X_train = pd.read_csv('datasets/processed/X_train.csv')
    X_val = pd.read_csv('datasets/processed/X_val.csv')
    y_train = pd.read_csv('datasets/processed/y_train.csv').values.ravel()
    y_val = pd.read_csv('datasets/processed/y_val.csv').values.ravel()
    
    return X_train, X_val, y_train, y_val

def train_and_evaluate(X_train, X_val, y_train, y_val):
    models = {
        "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }
    
    results = []
    best_model = None
    best_f1 = 0.0
    
    print(f"\nTraining {len(models)} models...")
    print("-" * 60)
    print(f"{'Model':<30} | {'Accuracy':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("-" * 60)
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        # Evaluate
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        roc = roc_auc_score(y_val, y_prob)
        
        print(f"{name:<30} | {acc:.4f}     | {f1:.4f}     | {roc:.4f}")
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "F1-Score": f1,
            "ROC-AUC": roc,
            "Object": model
        })
        
        # Track best model based on F1-Score
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            
    print("-" * 60)
    return results, best_model

def save_best_model(model):
    print(f"\nSaving best model: {type(model).__name__}")
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to 'models/best_model.pkl'")

def main():
    X_train, X_val, y_train, y_val = load_data()
    results, best_model = train_and_evaluate(X_train, X_val, y_train, y_val)
    save_best_model(best_model)
    
    # Save results to CSV for reporting
    results_df = pd.DataFrame(results).drop(columns=['Object'])
    results_df.to_csv('datasets/processed/model_results.csv', index=False)
    print("Results saved to 'datasets/processed/model_results.csv'")

if __name__ == "__main__":
    main()
