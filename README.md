# Cardiovascular Disease Prediction — Clinical ML Diagnostic System

> Multi-model machine learning system for cardiovascular risk classification with SHAP explainability, deployed on Streamlit.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)](https://streamlit.io)
[![Live Demo](https://img.shields.io/badge/Demo-Live-brightgreen)](https://cardiovascular-disease-prediction-ajmznkpqhaewp2xwdmhcgc.streamlit.app/)

**Live application:** https://cardiovascular-disease-prediction-ajmznkpqhaewp2xwdmhcgc.streamlit.app/

---

## Overview

This project develops an intelligent cardiovascular risk prediction system using advanced machine learning techniques. The application provides an intuitive interface for clinicians and users to obtain a personalized cardiovascular risk assessment based on clinical parameters.

---

## Architecture

```
Raw Clinical Data
    |
Preprocessing (imputation, encoding, normalization)
    |
Model Training (5 algorithms, cross-validation)
    |
Model Selection & Serialization
    |
Streamlit Application
    |
Real-time Prediction + SHAP Explainability + Recommendations
```

---

## Models Implemented

| Model | Notes |
|-------|-------|
| Random Forest | Primary model — feature importance via SHAP |
| XGBoost | Gradient boosting, strong performance |
| LightGBM | Fast training, large dataset efficiency |
| Logistic Regression | Interpretable baseline |
| Decision Tree | Visual explainability |

---

## Key Features

- Interactive prediction form with real-time validation
- SHAP-based feature importance for each individual prediction
- ROC curves and confusion matrices across all models
- Correlation matrix and data distribution visualizations
- Personalized clinical recommendations per risk level

---

## Installation

```bash
git clone https://github.com/KalsoumDS/cardiovascular-disease-prediction.git
cd cardiovascular-disease-prediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Technologies

- Python 3.10+, Scikit-learn, XGBoost, LightGBM
- SHAP, Plotly, Streamlit, pandas, NumPy

---

## Author

Oumou Kaltoum Sall — Data Scientist & ML Engineer  
[Portfolio](https://luxury-sunshine-073627.netlify.app) · [LinkedIn](https://linkedin.com/in/oumou-kaltoum-sall)
