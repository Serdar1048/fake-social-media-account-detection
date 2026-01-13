# 🕵️‍♂️ Social Media Anomaly Detection: Technical Depth Report

**Date:** January 13, 2026 
**Prepared By:** Serdar Dedebas

---

## 1. Introduction: Statistics Behind Digital Masks

Social media platforms are vast oceans of data where billions of users interact. However, swimming alongside organic users in this ocean are synthetic identities (fake/bot accounts) created for manipulation. The primary goal of this project is to detect whether a social media account is "human" or "software" based on statistical traces.

This report documents the technical architecture of the developed AI model, our discoveries on the data, feature engineering decisions, and the model's decision mechanism in detail. Our aim is not just to say "this account is fake," but to present **"why it is fake"** with mathematical proofs.

---

## 2. Data Discovery and Analysis (Exploratory Data Analysis)

Every successful machine learning project begins with understanding the data.

### 2.1. Data Source and Feature Pool 🗂️
The dataset used in the project was obtained from **[Kaggle: Fake/Authentic User Instagram](https://www.kaggle.com/datasets/krpurba/fakeauthentic-user-instagram)** to reflect real-world scenarios. The dataset contains 18 different features ranging from user profile information to content sharing habits.

These features are like digital fingerprints revealing the characteristics of an account:

**A. Account Profile Features**
*   **`pos` (Number of Posts):** Total number of posts shared by the user. Bots usually have either very few (newly opened) or too many (spam) posts.
*   **`flg` (Following):** Number of people the user follows. Bots generally follow many people to gain followers (Follow-for-Follow).
*   **`flr` (Followers):** Number of people following the account.
*   **`bl` (Bio Length):** Character count of the profile description.
*   **`pic` (Profile Picture):** Whether there is a profile picture (1) or not (0). Most real users have a picture, whereas bots sometimes stay with the default avatar.
*   **`lin` (External Link):** Presence of a link in the bio. Spam accounts often contain ad or malicious links.

**B. Content and Sharing Features**
*   **`cl` (Caption Length):** Average character length of the text under posts.
*   **`cz` (Empty Caption Ratio):** Ratio of posts with very short descriptions (less than 3 characters). Bots usually don't write descriptions.
*   **`ni` (Non-Image Media):** Ratio of video or carousel posts. Bots generally upload simple images.
*   **`lt` (Location Tag):** Ratio of using location tags in posts.
*   **`hc` (Hashtag Count):** Average number of tags per post.
*   **`cs` (Cosine Similarity):** How similar the posts are to each other. Bots usually share the same description or image repeatedly (High score = Suspicious).
*   **`pi` (Posting Interval):** Average time between two posts (in hours). Impossible tightness and regularity are signs of a bot.

**C. Engagement and Keyword Features**
*   **`erl` / `erc` (Engagement Rates):** Ratio of likes and comments to the number of followers. Bots have many followers but generally very low interaction (like/comment).
*   **`pr` (Promotional Keywords):** Frequency of use of ad-content words like "Giveaway", "repost", "contest".
*   **`fo` (Follower Hunter Keywords):** Frequency of use of tags aimed at gaining followers like "follow me", "f4f", "like".
*   **`class` (Class):** Our target variable. `f` (fake) or `r` (real).

### 2.2. Class Distribution and Balance
For the model not to be biased, it is critical that the dataset is balanced. In our analysis, we saw that "Real" and "Fake" accounts are distributed in close proportions. This indicates that the `Accuracy` metric can be a reliable performance indicator.

![Class Distribution](class_distribution.png)
*(Chart 1: Numerical distribution of Real and Fake accounts in the dataset. A balanced structure ensures the model learns both classes with equal weight.)*

### 2.3. Correlation Matrix of Features
We examined the correlation matrix to understand which features are related to each other (Multicollinearity) and which have a strong connection with the target variable (`class`).

*   **Observation:** There are expected positive correlations between some features (e.g., follower count and average likes). However, what matters for us is the relationships with the `class` (target) variable.
*   **Interpretation:** Dark areas represent strong relationships. Especially the relationship of derived ratios (e.g., follower/following ratio) with the target variable is noteworthy.

![Correlation Heatmap](correlation_heatmap.png)
*(Chart 2: Correlation map between features. Colors show the direction and intensity of the relationship.)*

### 2.4. Deep Dive: "Followers" and "Engagement"
The biggest vulnerability that reveals fake accounts is behavioral inconsistencies.

*   **Followers:** In real accounts, the follower count usually follows a logarithmic distribution (few people have many followers). In fake accounts, this distribution is more synthetic.
    ![Followers vs Class](followers_vs_class.png)
    *(Chart 3: Distribution of follower counts by class (Boxplot). Note how outliers and median differences separate the classes.)*

*   **Engagement:** A bot account might follow thousands of people, but it is hard to create "real" engagement (comment/like ratio). The chart below shows the relationship between Like Rate (`erl`) and Comment Rate (`erc`). Fake accounts generally cluster in low engagement areas, while real accounts spread to a wider area.
    ![Engagement Scatter](engagement_scatter.png)
    *(Chart 4: Scatter plot of Like and Comment rates. The separation between clusters proves the model can decide using these two features.)*

---

## 3. Feature Engineering

Raw data is not always sufficient for a model. It is necessary to translate the data into a language the model can "understand".

### 3.1. Derived and Selected Features
80% of the model's success depends on the selection of the right features.
*   **`ratio` (Follower / Following Ratio):** This is the "Star Feature" of the project.
    *   *Logic:* A bot usually does "Follow for Follow" to gain followers. This pulls the ratio to 1.0. In a real phenomenon, this ratio can go up to 1000s.
*   **`ni` (Non-Image Ratio):** Ratio of video or image-less content. Bot software likes text-based interaction; image processing is costly.

### 3.2. Dropped Features 🗑️
When taking the model out of the lab and adapting it to the real world, we had to make a trade-off between "Theoretical Maximum Accuracy" and "Practical Usability". The following features, although effective in training the model, were removed from the system because it is impossible or very difficult for the end-user to calculate them manually:

*   **`pi` (Posting Interval):** A complex metric like "Standard deviation between the posting hours of the last 30 posts". It takes minutes for a human to calculate this.
*   **`erl` & `erc` (Engagement Rate Likes/Comments):** Like and Comment rates. These are dynamic and change instantly. Asking the user to calculate the "Total Likes / Total Posts / Follower Count" formula is a UX (User Experience) killer.
*   **`pr` (Promotional Keywords):** How many promotional words like "#giveaway", "#repost" appear in the content. The user would need to perform NLP (Natural Language Processing) to detect this.
*   **`fo` (Follower Hunter Keywords):** Frequency of usage of tags like "#follow4follow", "#like4like". Again, manual detection is very difficult.

**Decision:** Although removing these features took away an insignificant amount (around 0.2-0.3%) from the model's accuracy, it ensured the system is usable by everyone within 10 seconds. A "Fast and Good" model is superior to a "Perfect but Unusable" model.

![Feature Importance](feature_importance.png)
*(Chart 5: Importance ranking showing which feature the model relies on most when deciding. Features at the top of the list are fake account hunters.)*

---

## 4. Model Selection and Performance Comparison 🌲 vs 🚀

Within the scope of the project, four different powerful algorithms were brought to the "Arena" and raced fiercely. Here is the performance report card of the models on the test set:

| Model | Accuracy | F1-Score | ROC-AUC | Status |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **0.8249** | **0.8177** | **0.9087** | 🥇 (Score Leader) |
| **Random Forest** | 0.8222 | 0.8133 | 0.9038 | 🌲 **(Selected Model)** |
| **Neural Network (MLP)** | 0.8203 | 0.8093 | 0.9011 | Contender |
| **Logistic Regression** | 0.7830 | 0.7752 | 0.8600 | Baseline |

### Why Random Forest and Not XGBoost? 🤔
Looking at the table, it is seen that XGBoost is mathematically superior (by a difference of 0.2%). However, engineering decisions are not made solely based on raw scores. The critical reasons why **Random Forest** was selected as the main model in our project are:

1.  **Robustness:** XGBoost follows an aggressive (boosting) method to minimize errors. In social media data, the probability of labels (real/fake) being incorrect is high. XGBoost might "memorize" (overfit) these incorrect data and artificially inflate the score. Random Forest, on the other hand, is more resistant and reliable against such noise because it looks at the majority vote with the "Bagging" method.
2.  **Generalizability:** A difference of 0.2% is a negligible gain in real life. In contrast, Random Forest tends to work more stably on new and unseen data types.
3.  **Explainability:** The aim of our project is to answer the question "why is it fake?". Random Forest's decision mechanism is more transparent and traceable thanks to its tree structure.

**Optimization Process (GridSearchCV):**
The selected Random Forest model was not left in its raw state, but evolved by testing 72 different parameter combinations.
*   **`n_estimators`: 200** (Number of trees doubled for a more stable decision)
*   **`min_samples_split`: 5** (Filter to prevent overfitting)
We didn't leave the model with default settings. We found the "Genetic Codes" that gave the best result by testing 72 different combinations with the `GridSearchCV` technique. Here are the champion model's parameters:

*   **`n_estimators` (Number of Trees): 200** - 200 different decision trees work simultaneously in the system. (Standard is 100, we doubled it).
*   **`max_depth` (Depth): None** - We allowed the trees to deepen unlimitedly, so they could catch even the finest detail.
*   **`min_samples_split`: 5** - We required at least 5 data points for a branch to split into two (To prevent overfitting).
*   **`min_samples_leaf`: 2** - We mandated at least 2 accounts in each leaf.

---

## 5. Performance Evaluation: How is Our Report Card? 📊

We put the model through a tough exam on the "Test Set" (data it has never seen). The results prove that we have built a reliable system:

### 5.1. Metrics and Meanings
Just "Accuracy" alone is not enough. Here is the detailed report card of the model:

| Metric | Value | Meaning |
| :--- | :--- | :--- |
| **Accuracy** | **%83.00** | The model correctly identified 83 out of every 100 accounts. |
| **F1-Score** | **0.83** | The score of "knowing both fake and real balanced". (Out of 1.0). |
| **Precision** | **0.83** | 83% of the accounts we called "Fake" were actually fake. (False alarm is low). |
| **Recall** | **0.83** | We were able to catch 83% of all fake accounts in the market. (Few missed fish). |

**Comment:** An accuracy rate of 83% is quite successful in a noisy environment (where there is mislabeled data) like social media. The F1 score being 0.83 shows that the model does not try to be cunning like "Just find the real ones, ignore the fakes", but recognizes both classes with equal success.

### 5.2. Confusion Matrix
Let's look at the matrix just to see where we made mistakes:
*   **False Positive (False Alarm):** Our rate of calling a real person "Fake" is low.
*   **False Negative (Missed Fish):** Our rate of mistaking a fake person for "Real" is low.

The matrix below visualizes the stability of the model. High density on the diagonal is proof of success.

![Confusion Matrix](confusion_matrix.png)
*(Chart 6: Comparison of predictions on the test set with actual values.)*

### 5.3. ROC Curve and AUC
The ROC curve shows the discrimination power of the model. The closer the curve is to the top left corner, the more perfect the model is. Our model proved to be far superior to a random predictor by performing well above the line.

**AUC (Area Under Curve) = 0.91** value has the following practical meaning in our project:
1.  **Discrimination Power:** When given a random **real** and a **fake** account, the probability of the model correctly distinguishing the fake account with a higher risk score than the real account is **%91**.
2.  **Reliability:** Compared to a random predictor with a 50% chance of success (coin toss); the score of **0.91** proves that the model performs in the **"Excellent"** category. This shows that the model can clearly separate fake accounts even in complex and noisy data like social media.

![ROC Curve](roc_curve.png)
*(Chart 7: True Positive Rate vs False Positive Rate. The area under the curve (AUC) summarizes the model's superior success with a value of 0.91.)*

---

## 6. Hybrid Decision Architecture: Statistics + Expert Rule

Our dataset was from 2017, and concepts like "Blue Tick" or "Highlights" were not as common/meaningful back then. The model **does not know** what a Blue Tick is.

Therefore, we designed a **Hybrid Architecture** that blends pure Machine Learning output (a risk score between 0 and 1) with expert rules (Rule-Based).

This architecture is based on the **"Feed-Forward Risk Calculation with Feedback Penalty"** principle. The system works in two stages:

**Stage 1: Statistical Risk Prediction (The Mathematical Core)**
In the first step, the Random Forest model looks only at the numerical data of the account (follower count, following count, posting frequency, etc.). At this stage, the model does not care "who" the account is, it only examines the pattern formed by the numbers. The model produces a raw **"AI Risk Score"** between 0 and 1 (Example: 0.95 - High Probability Fake).

**Stage 2: Contextual Reliability Discount**
In the second stage, "social proofs" that are not in the model's training set but are accepted as reliability signals in the real world come into play. These signals (Blue Tick, Carousel Post, Highlight Stories) "penalize", i.e., pull down, the calculated risk score.

This process is not a simple subtraction, but a **multiplicative decay** process:

$$ FinalRisk = RawRisk \\times (1 - TrustFactor_1) \\times (1 - TrustFactor_2) ... $$

*   **Verified Badge (Blue Tick):** It is the strongest trust signal. It directly dampens the risk score by 15%.
*   **Carousel Post:** Sharing multiple media in a single post (album with right swipe) is an action that simple bot software usually cannot do. This "human effort" indicator reduces the risk score by 15%.
*   **Highlights:** Creating a story archive is a complex and costly process for bot software. This feature is a strong indicator that there is a real human behind the account and reduces the risk by 5%.

**Result:**
Even if the model mathematically says "Fake" by 90% for an account, if that account is "Blue Ticked" and "Has Highlights", the risk score is dramatically reduced (%90 -> %72 -> %68) and pulled to the safe zone (Real Account). This approach minimizes the "False Positive" (False Alarm) rate by stretching the strict rules of artificial intelligence with human intuition.

Thanks to this architecture:
1.  **Adaptation:** New rules (e.g., a new type of badge) can be added to the system without retraining the model.
2.  **Trust:** "Human Intelligence" steps in where the dataset falls short.

---

## 7. Conclusion and Comment

This project is more than a "Classification" problem; it is a digital detective work.

*   An account that looks "normal" to the naked eye can be instantly caught by the model thanks to the **Follower/Following ratio** and **Content Type Imbalance** (`ni`).
*   The eliminations we made during the **Feature Selection** phase ensured that the model is both light and applicable in the field.
*   The **Hybrid Architecture** we designed covered the "blind spots" of pure artificial intelligence (lack of context) with simple but effective rules.

As a result, the resulting system is a data-driven, statistical, explainable, and high-accuracy anomaly detection engine.

---

