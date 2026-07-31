# 🛌 Sleep Health & Lifestyle Risk Analyzer

An interactive machine learning app that predicts sleep disorder risk, 
estimates sleep quality, and segments users into lifestyle types based 
on daily habits — with live, explainable predictions.

## Features
- Predicts sleep disorder risk (None / Insomnia / Sleep Apnea)
- Estimates sleep quality score (1-10)
- Segments users into lifestyle types via clustering
- Interactive Streamlit app with live slider-based predictions
- Feature importance shown for full explainability

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (Random Forest Classifier & Regressor, K-Means Clustering)
- Streamlit + Plotly for the interactive app

## Dataset
[Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset) (Kaggle)

## Live Demo
https://sleep-health-lifestyle-risk-analyzer-migjubphrzesutdjhh7dti.streamlit.app/

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
- `Sleep_Health.ipynb` — data cleaning, EDA, and model training
- `app.py` — Streamlit application
- `*.pkl` — saved trained models and encoders
