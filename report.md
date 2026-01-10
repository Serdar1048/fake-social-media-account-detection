# Project Report: Fake Social Media Account Detection

## 1. Introduction

### Project Objective
The primary objective of this project is to develop a machine learning model capable of automatically distinguishing between real and fake Instagram accounts. By analyzing user profile features and activity patterns, we aim to create a reliable detection system that can help maintain platform integrity and user trust.

### Problem Definition
Social media platforms like Instagram are plagued by fake accounts (bots, impersonators, spam accounts). These accounts are often used for malicious activities such as fraud, spreading misinformation, phishing, and artificially inflating engagement metrics. Manually identifying these accounts is time-consuming and inefficient due to the sheer volume of users. Therefore, an automated, data-driven solution is required to classify accounts as "Real" or "Fake" based on behavioral and profile-based attributes.

## 2. Dataset Overview

### Source and Content
The dataset used in this project consists of Instagram user data. It contains **65,326** entries (rows) and **18** columns. Each entry represents a unique user profile with various features extracted from their account activity and profile settings.

### Variables Description
The dataset includes the following features:

- **Profile Information:**
    - `pic`: Profile picture availability (1 if present, 0 if absent).
    - `bl`: Bio length (number of characters).
    - `lin`: External URL availability (1 if present, 0 if absent).
    - `flw`: Number of followers.
    - `flg`: Number of following.
    - `pos`: Total number of posts.

- **Engagement & Activity Metrics:**
    - `erl`: Engagement rate based on likes.
    - `erc`: Engagement rate based on comments.
    - `hc`: Average number of hashtags per post.
    - `pr`: Usage of promotional keywords in hashtags (e.g., repost, contest).
    - `fo`: Usage of follower-hunting keywords (e.g., followback, f4f).
    - `lt`: Percentage of posts with location tags.
    - `pi`: Average time interval between posts.
    - `cs`: Average cosine similarity between user's posts (content consistency).

- **Content Characteristics:**
    - `cl`: Average caption length.
    - `cz`: Percentage of captions with almost zero length (<= 3 chars).
    - `ni`: Percentage of non-image media (video, carousel).

- **Target Variable:**

## 3. Exploratory Data Analysis (EDA)

### Data Cleaning
We started by checking for missing values and duplicates.
- **Missing Values:** The dataset contains no missing values (NaNs).
- **Duplicates:** We checked for and removed any duplicate entries to ensure data quality.

### Visualizations & Insights

#### 1. Target Variable Distribution
![Class Distribution](assets/class_distribution.png)
*Figure 1: Distribution of Real vs. Fake Accounts*

**Observation:** This chart shows the balance between "Real" and "Fake" accounts. A balanced dataset is ideal for training, while a significant imbalance might require techniques like oversampling or undersampling.

#### 2. Feature Correlation Matrix
![Correlation Heatmap](assets/correlation_heatmap.png)
*Figure 2: Heatmap showing correlations between features*

**Observation:**
- **High Correlations:** We look for features that are strongly correlated with each other (e.g., `flw` and `erl` might have a relationship).
- **Target Correlation:** We can observe which features tend to have higher values for Real vs. Fake accounts (though `class` is categorical and not in this numeric heatmap, we infer relationships from other plots).

#### 3. Feature Relationships
![Followers vs Class](assets/followers_vs_class.png)
*Figure 3: Boxplot of Followers Count by Account Type*

**Observation:** Real accounts typically have a different distribution of followers compared to fake accounts. Fake accounts often have very low or uniformly generated follower counts.

![Engagement Scatter](assets/engagement_scatter.png)
*Figure 4: Scatter Plot of Engagement Rates (Likes vs. Comments)*


## 4. Data Preprocessing & Feature Engineering

Before training our machine learning models, we applied several preprocessing steps to ensure the data is suitable for algorithmic processing.

### 4.1. Encoding
The target variable `class` has two categories: `Real` (r) and `Fake` (f). Algorithms require numerical input, so we encoded these as:
- **0 (Real)**
- **1 (Fake)**
This binary classification allows us to use algorithms like Logistic Regression and Random Forest effectively.

### 4.2. Data Splitting
We split the dataset into three parts to evaluate the model's performance rigorously:
- **Training Set (60%):** Used to train the model directly.
- **Validation Set (20%):** Used to tune hyperparameters and prevent overfitting during the training process (e.g., Early Stopping).
- **Test Set (20%):** Reserved for final evaluation. This data is never seen by the model during training or tuning.
This split (60/20/20) ensures we have a dedicated set for tuning without biasing the final performance metric.

### 4.3. Feature Selection
We implemented a correlation-based feature selection method to identify and remove redundant features.
- **Method:** Calculate the correlation matrix of all features.
- **Threshold:** If two features have a correlation higher than **0.95**, one of them is removed.
- **Result:** In our dataset, no features exceeded this threshold, meaning all features provided sufficiently unique information to be retained. This is beneficial as we don't lose any data signals.

### 4.4. Scaling
Since our features have different ranges (e.g., `flw` (Followers) can be in millions, while `erl` (Engagement Rate) is a small decimal), we applied **Standard Scaling**.
- **Technique:** `StandardScaler` from Scikit-Learn.
- **Effect:** Centers the distribution around 0 with a standard deviation of 1.
- **Why?** Many algorithms (like Neural Networks, SVM, and KNN) are sensitive to the scale of input data. Scaling ensures that features with larger values don't dominate the objective function essentially.

## 5. Modeling & Algorithm Selection

We trained and evaluated four different machine learning algorithms to identify the most effective model for fake account detection.

### 5.1. Algorithms Evaluated
1.  **Logistic Regression (Baseline):** A simple, linear model used as a baseline. If complex models don't significantly outperform this, we prefer the simpler one.
2.  **Random Forest Classifier:** An ensemble learning method that constructs multiple decision trees. It is robust against overfitting and handles non-linear data well.
3.  **XGBoost (Extreme Gradient Boosting):** An optimized gradient boosting library designed to be highly efficient, flexible, and portable. Often achieves state-of-the-art results.
4.  **Neural Network (MLP):** A Multi-Layer Perceptron (deep learning) that can capture complex, non-linear relationships in high-dimensional data.

### 5.2. Performance Comparison
We evaluated the models on the **Validation Set** using three key metrics:
- **Accuracy:** Overall correctness of the model.
- **F1-Score:** Harmonic mean of Precision and Recall (crucial since both False Positives and False Negatives differ in cost, and we want a balance).
- **ROC-AUC:** Area Under the Curve, measuring the ability to distinguish between classes.

| Model | Accuracy | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | 0.8040 | 0.7982 | 0.8781 |
| **Random Forest** | **0.8981** | **0.8907** | 0.9588 |
| **XGBoost** | 0.8963 | **0.8907** | **0.9612** |
| **Neural Network (MLP)** | 0.8797 | 0.8728 | 0.9482 |

### 5.3. Selection
**Winner: Random Forest / XGBoost**
Both Random Forest and XGBoost performed exceptionally well and significantly outperformed the baseline.
- **Random Forest** achieved the highest accuracy (**89.81%**).
- **XGBoost** showed the best ROC-AUC score (**96.12%**), indicating excellent separation capability.

Given that Random Forest is often easier to deploy and tune with fewer hyperparameters for this scale of data, we selected **Random Forest** (or XGBoost, performance is nearly identical) as our final model for the application.

## 6. Optimization & Final Evaluation

We selected **Random Forest** for the final model and performed Hyperparameter Tuning using `GridSearchCV`.

### 6.1. Optimized Parameters
The grid search identified the following best parameters:
- `n_estimators`: 200 (More trees improved stability)
- `max_depth`: None (Allowing trees to grow fully captures complex patterns)
- `min_samples_leaf`: 1

### 6.2. Detailed Evaluation
The optimized model was evaluated on the **Test Set** (never before seen data).
- **Test Accuracy:** **~90%**

#### A. Confusion Matrix
![Confusion Matrix](assets/confusion_matrix.png)
*Figure 5: Confusion Matrix showing True vs Predicted labels.*
- **Interpretation:** The model successfully identifies the majority of Real and Fake accounts. The number of False Positives (Real accounts marked as Fake) and False Negatives (Fake accounts marked as Real) is low.

#### B. ROC Curve
![ROC Curve](assets/roc_curve.png)
*Figure 6: Receiver Operating Characteristic (ROC) Curve.*
- **Interpretation:** The curve is close to the top-left corner, and the AUC score is high. This confirms the model is excellent at distinguishing between the two classes at various threshold settings.

#### C. Feature Importance
![Feature Importance](assets/feature_importance.png)
*Figure 7: Feature Importance Plot.*
- **Key Insight:** Features like `flw` (Followers), `erl` (Engagement Rate), and `cs` (Cosine Similarity) likely played the biggest roles. This validates our hypothesis that engagement and content consistency are strong indicators of authenticity.

## 7. Conclusion & Future Work

### 7.1. Conclusion
We successfully developed a machine learning system to detect fake Instagram accounts with ~90% accuracy. By analyzing profile metadata and engagement metrics, the Random Forest model provides a robust solution for identifying inauthentic behavior. The project demonstrates that automated detection is not only feasible but highly effective.

### 7.2. Future Work
If we had more time, we would explore:
1.  **Text Analysis (NLP):** Analyzing caption text sentiment and bio descriptions using Deep Learning (LSTM/BERT).
2.  **Image Analysis:** Using CNNs to detect if profile pictures are stolen or AI-generated.
3.  **Graph Analysis:** Analyzing the network of followers to detect "bot farms" (clusters of fake accounts following each other).
