# Credit Risk Prediction System

An end-to-end machine learning project that predicts the repayment risk of loan applicants using the Home Credit dataset.

## 🚀 Features

- Exploratory Data Analysis
- Data cleaning and preprocessing
- Feature engineering
- Class imbalance handling using Random Oversampling
- Logistic Regression, Decision Tree, Random Forest
- Artificial Neural Network (ANN)
- Model comparison and threshold analysis
- SHAP-based explainability
- Streamlit deployment

## 📊 Model Performance

| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 68.97% | **67.43%** | 25.97% | **0.7482** |
| Random Forest | **78.45%** | 49.93% | **27.22%** | 0.7319 |
| ANN | 70.49% | 63.22% | 25.70% | 0.7286 |
| Decision Tree | 67.15% | 64.13% | 23.97% | 0.7022 |

**Selected Model:** Logistic Regression

Selected based on its higher recall and ROC-AUC, which are important for identifying potentially high-risk applicants.

## 🔍 Explainability

SHAP was used to identify the features contributing most to credit-risk predictions.

## 🌐 Streamlit App

The application accepts applicant information and provides:

- Predicted risk probability
- Low Risk / High Risk classification

Run locally:

```bash
pip install -r requirements.txt
streamlit run app/app.py
🛠️ Tech Stack
Python · Pandas · NumPy · Scikit-learn · TensorFlow · SHAP · Streamlit
📁 Structure
Credit_Risk_Prediction/
├── app/
├── data/
├── models/
├── Credit_Risk_Project.ipynb
├── README.md
└── requirements.txt
⚠️ Disclaimer
For educational and portfolio purposes only. The model should not be used as the sole basis for real-world lending decisions.