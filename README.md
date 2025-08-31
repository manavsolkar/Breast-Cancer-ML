# Breast Cancer Detection Project

This project predicts whether a breast tumor is **Malignant (cancerous)** or **Benign (non-cancerous)** using the Breast Cancer Wisconsin dataset.

## Features
- Exploratory Data Analysis (EDA)
- Machine Learning Model (RandomForest Classifier)
- Evaluation: Confusion Matrix, ROC Curve, Feature Importance
- Flask Web App for predictions
- Report folder with metrics and plots

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run Jupyter Notebook for analysis: `jupyter notebook breast_cancer_expanded_notebook.ipynb`
3. Launch web app: `python app.py`
4. Open in browser: `http://127.0.0.1:5000/`

## Project Structure
```
breast_cancer_ml_project_full/
│── breast_cancer_expanded_notebook.ipynb
│── app.py
│── requirements.txt
│── .gitignore
│── README.md
│── report/
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── feature_importance.png
    ├── metrics.json
```

## Dataset
The dataset is from `sklearn.datasets.load_breast_cancer`.

---
✨ Developed for quick ML deployment & portfolio use.
