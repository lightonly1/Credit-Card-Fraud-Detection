# Credit Card Fraud Detection
## Project Overview
This project focuses on identifying fraudulent credit card transactions using machine learning techniques.  
The dataset is highly imbalanced, with legitimate transactions vastly outnumbering fraudulent ones.  
The objective is to develop models that can accurately detect fraud while minimizing false positives.

---

## Objectives
- Explore and understand the dataset structure  
- Handle data imbalance using oversampling methods  
- Build and compare multiple classification models  
- Evaluate performance using precision, recall, F1-score, and ROC-AUC metrics  

---

## Dataset
- **Source:** [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Transactions:** 284,807  
- **Fraud Cases:** 492 (0.172%)  
- **Features:** 30 anonymized variables (V1–V28 obtained through PCA)  
- **Target:** `Class` — 1 indicates fraud, 0 indicates a valid transaction  

---

## Methodology
1. **Data Preparation**
   - Loaded the dataset and performed basic exploration  
   - Checked for null values and outliers  
   - Scaled numerical features using StandardScaler  
2. **Exploratory Data Analysis**
   - Analyzed transaction distribution over time and amount  
   - Examined correlations and class imbalance  
3. **Resampling**
   - Applied Synthetic Minority Over-sampling Technique (SMOTE) to balance the classes  
4. **Model Development**
   - Trained Logistic Regression, Decision Tree, and Random Forest models  
5. **Model Evaluation**
   - Used confusion matrix, precision, recall, F1-score, and ROC-AUC to assess model performance  

---

## Model Performance

| Model | ROC-AUC | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) | Accuracy |
|--------|----------|-------------------|----------------|------------------|-----------|
| Logistic Regression | **0.9769** | 0.08 | **0.93** | 0.15 | 0.98 |
| Decision Tree | 0.9071 | 0.40 | 0.82 | 0.54 | 1.00 |
| Random Forest | 0.9728 | **0.83** | 0.87 | **0.85** | 1.00 |

---

## Observations
- Logistic Regression achieved the highest ROC-AUC score of **0.977**, showing strong ability to distinguish between fraud and non-fraud cases.  
- Random Forest also performed well, maintaining high precision and recall.  
- The Decision Tree model worked as a baseline but was prone to overfitting.  
- Due to the class imbalance, accuracy is not a reliable metric; recall and ROC-AUC provide a clearer picture of performance.  

---

## Technologies Used
- **Language:** Python  
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn  
