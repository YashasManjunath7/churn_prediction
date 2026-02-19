# 📡 Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> Predicting which telecom customers are likely to churn using machine learning — with full model explainability using SHAP.

---

## 📌 Problem Statement

Customer churn is one of the biggest challenges in the telecom industry. Acquiring a new customer costs **5x more** than retaining an existing one. This project builds a machine learning pipeline to identify at-risk customers early, enabling targeted retention strategies.

---

## 📂 Dataset

- **Source:** [Telco Customer Churn — IBM Dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
- **Size:** 7,043 customers, 21 features
- **Target:** `Churn` — whether the customer left within the last month
- **Class Imbalance:** ~26% churn vs ~74% no churn → handled using SMOTE

---

## 🔍 Key Insights from EDA

- Customers with **month-to-month contracts** churn at a significantly higher rate than those on annual contracts
- **Higher monthly charges** are strongly associated with churn
- Customers with **lower tenure (< 12 months)** are the most likely to leave
- Customers without **online security or tech support** show higher churn rates

---

## ⚙️ Project Pipeline

```
Raw Data → EDA & Cleaning → Feature Engineering → Train/Test Split
→ SMOTE (handle imbalance) → Model Training → Evaluation → SHAP Explainability
```

---

## 🧠 Models Trained

| Model | AUC-ROC | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---|---|---|---|---|
| Logistic Regression | ~0.84 | ~0.63 | ~0.77 | ~0.69 |
| Random Forest | ~0.85 | ~0.66 | ~0.75 | ~0.70 |
| **XGBoost** | **~0.86** | **~0.68** | **~0.78** | **~0.73** |

✅ **XGBoost selected as the final model** based on the highest AUC-ROC score.

> Note: Metrics are reported on the held-out test set (20% split, stratified).

---

## 📊 Results & Visualizations

### ROC Curve Comparison
![ROC Curve](download (47).png)

### SHAP Feature Importance
![SHAP Summary](download (48).png)

**Top factors driving churn:**
- `tenure` — shorter tenure = higher churn risk
- `Contract_Two year` — long-term contracts strongly reduce churn
- `MonthlyCharges` — higher charges increase churn likelihood
- `InternetService_Fiber optic` — fiber users churn more despite premium service

---

## 🏗️ Tech Stack

| Area | Tools |
|---|---|
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Modeling | Scikit-learn, XGBoost |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Environment | Google Colab |


---

## 💡 Business Impact

If this model were deployed by a telecom company with 100,000 customers:
- Identifying the top 20% at-risk customers with ~78% recall catches most churners early
- Even a **10% improvement in retention** of flagged customers could save millions in lost revenue
- SHAP explanations allow business teams to take **targeted action** (e.g., offer contract upgrades to month-to-month customers with high monthly charges)

---

## 🔮 Future Improvements

- Hyperparameter tuning with Optuna or GridSearchCV
- Add a Streamlit web app for real-time predictions
- Experiment with CatBoost and LightGBM
- Deploy on Streamlit Community Cloud or Hugging Face Spaces

---
