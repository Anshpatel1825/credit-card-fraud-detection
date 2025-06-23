# Credit Card Fraud Detection Project
# Overview
- This project focuses on detecting fraudulent credit card transactions using machine learning techniques. With extreme class imbalance (only ~0.17% fraud cases), the goal was to accurately identify frauds while minimizing false negatives and false positives.

#  Key Objectives
- Handle extreme class imbalance effectively.
- Build and evaluate multiple ML models.
- Compare performance via ROC-AUC, Confusion Matrix, and Classification Reports.
- Identify top fraud indicators via EDA and feature importance.

# Dataset Summary
- Total Records: ~284,807
- Fraudulent Transactions: ~492
- Features: Time, Amount, and V1 to V28 (PCA transformed)

# Exploratory Data Analysis (EDA)
### 1> Fraud vs Non-Fraud Count Plot
![Fraud vs Non-Fraud Count Plot](images/fraud_roc_curve.png)
- 99.83% transactions are non-fraud, only 0.17% are fraud.
- Extreme imbalance → model needs imbalance handling (SMOTE, class weights, etc.).
  
### 2> Amount by Class (Box Plot)
![Amount by Class (Box Plot)](images/fraud_roc_curve.png)
- Fraud avg ₹122 > Non-fraud avg ₹88
- Fraud median ₹9 < Non-fraud median ₹22
- Outliers in both → right-skewed distribution → scaling needed.
  
### 3> Hourly Transaction Distribution
![Hourly Transaction Distribution](images/fraud_roc_curve.png)
- Non-fraud peaks in 8 AM–6 PM (working hours).
- Fraud scattered, slightly more in night (0–6 AM).
- Log scale helps visualize rare fraud patterns.
- 
### 4> Correlation Heatmap (Masked)
![Correlation Heatmap (Masked)](images/fraud_roc_curve.png)
- Top negative: V17, V14, V12, V10 (fraud ↑ as value ↓)
- Top positive: V11, V4, V2 (fraud ↑ as value ↑)
- Time & Amount have near-zero correlation with fraud.
- 
### 5> Target ‘Class’ Correlation Focus
![Target ‘Class’ Correlation Focus](images/fraud_roc_curve.png)
- Strong indicators: V14, V12, V10, V17 (clear fraud separators)
- Moderate: V11, V4, V2
- Weak/irrelevant: V24, V13
- 
### 6> KDE Plots for V10, V12, V14, V17
- hese features show clear separation between fraud and non-fraud classes.

# Data Preprocessing
- Feature and target separation (X, y)
- Handled imbalance using SMOTE (for Logistic Regression only)
- Feature scaling using StandardScaler (for Logistic Regression only)

# Machine Learning Models

## 1>Logistic Regression
SMOTE applied + scaling
Model trained on balanced & scaled data
![Logistic Regression](https://link-to-your-image.com/image.png)



#### Confusion Matrix:

- TP (Fraud correctly): 108
- FN (Fraud missed): 15
- FP (False alarm): 1221
- TN (Correct non-fraud): 69858

#### ROC Curve
- AUC Score: 0.96
![Logistic Regression ROC Curve](https://link-to-your-image.com/image.png)

  
#### Insights
- Recall (fraud): 88% — low fraud leakage
- Precision (fraud): ~8% — needs improvement

## 2> Random Forest (with class_weight='balanced')
- Trained directly on imbalanced data with built-in balancing
![Random Forest](https://link-to-your-image.com/image.png)

#### Confusion Matrix:
- TP: 123
- FN: 0
- FP: 69190
- TN: 1889

#### ROC Curve
- AUC Score: 0.94
  ![Random Forest ROC Curve](https://link-to-your-image.com/image.png)

  
#### Insights
- 100% fraud detection (0 FN)
- Very high false positives (many legitimate transactions flagged)

#### Feature Importance
- Top features: V14, V10, V4, V12, V17


## 3> XGBoost (scale_pos_weight=100)
- Trained on original imbalanced data
  ![XGBoost](https://link-to-your-image.com/image.png)


#### Confusion Matrix:

- TP: 111
- FN: 12
- FP: 3260
- TN: 67819

#### ROC Curve
- AUC Score: 0.98
![ XGBoost ROC Curve](https://link-to-your-image.com/image.png)

  
#### Insights
- Recall (fraud): 90%
- Precision (fraud): ~3%
- Balanced performance — best compromise between fraud capture and false alarms

# Final Model Recommendation
- Use XGBoost for production:
- High AUC (0.98)
- Best balance of fraud recall and manageable false positives

# Tools & Libraries Used
- Python (Pandas, NumPy, Sklearn, XGBoost)
- Seaborn, Matplotlib
- Imbalanced-learn (SMOTE)

#  Conclusion
- With a highly imbalanced dataset, traditional models underperform. By using SMOTE, class_weight, and boosting techniques, we significantly improved fraud detection capabilities. The XGBoost model emerged as the most reliable for production deployment.


