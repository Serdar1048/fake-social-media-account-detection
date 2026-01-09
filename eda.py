import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="whitegrid")

# Ensure assets directory exists
if not os.path.exists('assets'):
    os.makedirs('assets')

def load_data():
    print("Loading data...")
    df = pd.read_csv("datasets/users.csv")
    print(f"Initial Shape: {df.shape}")
    return df

def clean_data(df):
    print("\n--- Data Cleaning ---")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Duplicates removed. New Shape: {df.shape}")
        
    # Check for missing values
    missing = df.isnull().sum().sum()
    print(f"Total missing values: {missing}")
    
    # Check for target variable consistency
    print(f"Target variable counts:\n{df['class'].value_counts()}")
    
    return df

def plot_target_distribution(df):
    print("\nPlotting Target Distribution...")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='class', data=df, palette='viridis')
    plt.title('Distribution of Real (r) vs Fake (f) Accounts')
    plt.xlabel('Account Type')
    plt.ylabel('Count')
    plt.savefig('assets/class_distribution.png')
    plt.close()
    print("Saved: assets/class_distribution.png")

def plot_correlation_heatmap(df):
    print("\nPlotting Correlation Heatmap...")
    
    # Create a copy and encode 'class' to numeric for correlation
    # Mapping: f (fake) -> 1, r (real) -> 0
    corr_df = df.copy()
    corr_df['is_fake'] = corr_df['class'].apply(lambda x: 1 if x == 'f' else 0)
    
    # Select numeric columns including the new 'is_fake'
    numeric_df = corr_df.select_dtypes(include=['number'])
    
    plt.figure(figsize=(14, 12))
    corr = numeric_df.corr()
    
    # Plot heatmap
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Matrix)
    plt.tight_layout()
    plt.savefig('assets/correlation_heatmap.png')
    plt.close()
    print("Saved: assets/correlation_heatmap.png")

def plot_feature_relationships(df):
    print("\nPlotting Feature Relationships...")
    
    # Boxplot: Followers (flw) vs Class
    # Since followers can be skewed, we might want to log scale or just show as is.
    # Let's do a basic boxplot first.
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='class', y='flw', data=df, palette='Set2')
    plt.title('Followers Count by Account Type')
    plt.yscale('log') # Log scale mainly because followers distribution is usually power law
    plt.ylabel('Followers (Log Scale)')
    plt.savefig('assets/followers_vs_class.png')
    plt.close()
    print("Saved: assets/followers_vs_class.png")
    
    # Scatter: Engagement Rate Likes (erl) vs Engagement Rate Comments (erc)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='erl', y='erc', hue='class', data=df, alpha=0.6, palette='deep')
    plt.title('Engagement Rate: Likes vs Comments')
    plt.savefig('assets/engagement_scatter.png')
    plt.close()
    print("Saved: assets/engagement_scatter.png")

def main():
    df = load_data()
    df = clean_data(df)
    
    plot_target_distribution(df)
    plot_correlation_heatmap(df)
    plot_feature_relationships(df)
    
    print("\nEDA Completed Successfully.")

if __name__ == "__main__":
    main()
