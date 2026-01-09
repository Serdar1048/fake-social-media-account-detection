import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

def load_and_clean_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Basic Cleaning (same as Day 2)
    df = df.drop_duplicates()
    print(f"Data shape after cleaning: {df.shape}")
    return df

def preprocess_data(df):
    print("Preprocessing data...")
    
    # 1. Encoding
    # Map 'class' to 0 (Real) and 1 (Fake)
    # Note: Day 2 EDA showed 'f' and 'r'. Let's assume 'f'=1 (Fake), 'r'=0 (Real)
    df['class'] = df['class'].map({'f': 1, 'r': 0})
    
    # Separating Features and Target
    X = df.drop('class', axis=1)
    y = df['class']
    
    # 2. Splitting
    # We need Train, Validation, and Test sets.
    # Approach:
    # 1. Split Data -> Train+Val (80%) and Test (20%)
    # 2. Split Train+Val -> Train (75% of 80% = 60% total) and Val (25% of 80% = 20% total)
    print("Splitting data (60% Train, 20% Val, 20% Test)...")
    
    # First split: Separate Test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Second split: Separate Validation set from Training set
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42) # 0.25 * 0.8 = 0.2
    
    # 3. Feature Selection
    # Fit only on X_train to avoid leakage
    print("Performing Feature Selection...")
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    if to_drop:
        print(f"Dropping correlated features: {to_drop}")
        X_train = X_train.drop(columns=to_drop)
        X_val = X_val.drop(columns=to_drop)
        X_test = X_test.drop(columns=to_drop)
    else:
        print("No highly correlated features found to drop.")

    # 4. Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    # Fit ONLY on Training data
    X_train_scaled = scaler.fit_transform(X_train)
    # Transform Val and Test
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns) # Columns might have changed due to drop
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler

def save_artifacts(X_train, X_val, X_test, y_train, y_val, y_test, scaler):
    print("Saving artifacts...")
    
    # Ensure directories exist
    os.makedirs('datasets/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Save Datasets
    X_train.to_csv('datasets/processed/X_train.csv', index=False)
    X_val.to_csv('datasets/processed/X_val.csv', index=False)
    X_test.to_csv('datasets/processed/X_test.csv', index=False)
    y_train.to_csv('datasets/processed/y_train.csv', index=False)
    y_val.to_csv('datasets/processed/y_val.csv', index=False)
    y_test.to_csv('datasets/processed/y_test.csv', index=False)
    
    # Save Scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    print("Datasets saved to 'datasets/processed/'")
    print("Scaler saved to 'models/scaler.pkl'")

def main():
    raw_data_path = "datasets/users.csv"
    
    df = load_and_clean_data(raw_data_path)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_data(df)
    save_artifacts(X_train, X_val, X_test, y_train, y_val, y_test, scaler)
    
    print("\nPreprocessing Completed Successfully.")

if __name__ == "__main__":
    main()
