import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
import os

def load_data():
    print("Loading data...")
    X_train = pd.read_csv('datasets/processed/X_train.csv')
    X_val = pd.read_csv('datasets/processed/X_val.csv')
    X_test = pd.read_csv('datasets/processed/X_test.csv')
    
    y_train = pd.read_csv('datasets/processed/y_train.csv').values.ravel()
    y_val = pd.read_csv('datasets/processed/y_val.csv').values.ravel()
    y_test = pd.read_csv('datasets/processed/y_test.csv').values.ravel()
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def optimize_random_forest(X_train, y_train):
    print("optimizing Random Forest with GridSearchCV...")
    
    # Define Parameter Grid
    # Keeping it simple for speed, but effective enough
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    # We use 3-fold CV on Training set
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                               cv=3, n_jobs=-1, verbose=2, scoring='f1')
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    print("\nEvaluating on Test Set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    
    # Create assets directory if not exists
    os.makedirs('assets', exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix (Test Set)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('assets/confusion_matrix.png')
    plt.close()
    print("Saved: assets/confusion_matrix.png")
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('assets/roc_curve.png')
    plt.close()
    print("Saved: assets/roc_curve.png")
    
    return acc, roc_auc

def plot_feature_importance(model, feature_names):
    print("Plotting Feature Importance...")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 8))
    plt.title("Feature Importance")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig('assets/feature_importance.png')
    plt.close()
    print("Saved: assets/feature_importance.png")

def save_final_model(model):
    print("\nSaving final model...")
    with open('models/final_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to 'models/final_model.pkl'")

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # Combine Train and Val for final optimization/training to use maximum data
    # (Since we used Val for model selection in Day 4, we can now use it for final tuning)
    X_full_train = pd.concat([X_train, X_val])
    y_full_train = np.concatenate([y_train, y_val])
    
    best_rf_model = optimize_random_forest(X_full_train, y_full_train)
    
    evaluate_model(best_rf_model, X_test, y_test)
    plot_feature_importance(best_rf_model, X_train.columns)
    
    save_final_model(best_rf_model)
    print("\nOptimization Completed Successfully.")

if __name__ == "__main__":
    main()
