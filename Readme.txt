# 🏥 Emergency Department Wait Time Prediction

A machine learning project to predict Emergency Department (ED) wait times (represented by the 'Score' metric) using historical hospital data, demonstrating a complete MLOps workflow from Databricks experimentation to model deployment.

## 🚀 Overview

This project focuses on building an accurate regression model to predict the average waiting time/efficiency score for Emergency Departments. The final model is a **Voting Regressor Ensemble**, which was selected as the champion model after rigorous testing and tracking with MLflow.

The entire process, including data loading, preprocessing, model training, and performance logging, was executed within a **Databricks** environment.

### Key Technologies Used

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Data & Compute** | Databricks | Unified data and compute platform for execution. |
| **Model Tracking** | **MLflow** | Experiment tracking, metric logging, and model packaging (MLflow Pyfunc). |
| **Code & Versioning** | GitHub | Source code control and deployment trigger. |
| **Libraries** | `scikit-learn`, `XGBoost`, `Pandas` | Preprocessing and ensemble model training. |

---

## 📊 MLflow Experimentation

The model training was tracked in the Databricks-native MLflow instance under the experiment name: `/Users/arakkalakhila@gmail.com/Final_ED_WaitTime_Project`.

**Best Performing Model:** `VotingRegressorEnsemble`

| Metric | Champion Model Result | Description |
| :--- | :--- | :--- |
| **R²** | *[Insert R² value from your best run]* | Coefficient of Determination (closeness of fit). |
| **RMSE** | *[Insert RMSE value from your best run]* | Root Mean Squared Error (prediction error magnitude). |
| **MAE** | *[Insert MAE value from your best run]* | Mean Absolute Error. |

### 🔗 View Full Experiment Results

All training runs, metrics, and model artifacts are logged in the Databricks MLflow UI.

> **[Click here to view the experiment runs in Databricks]** (<PASTE DATABRICKS EXPERIMENT URL HERE>)

---

## 💻 Getting Started (Local Setup)

To run the training code locally (outside of Databricks), you must recreate the environment and provide the data.

### 1. Prerequisites

* Python 3.8+
* The raw dataset file: `FinalDS.xlsx` must be placed in the project root directory.

### 2. Installation

Create a virtual environment and install dependencies. You can generate a `requirements.txt` from your environment using `pip freeze > requirements.txt`.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR venv\Scripts\activate  # On Windows

# Install the required packages
pip install -r requirements.txt