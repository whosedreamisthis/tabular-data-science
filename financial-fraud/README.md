# Fraud Detection Engine

A high-performance machine learning pipeline designed to detect fraudulent financial transactions. This project focuses on handling extreme class imbalance and implementing feature engineering that respects the operational constraints of real-time production environments.

## Project Overview
This pipeline employs a `RandomForestClassifier` integrated with `SMOTE` (Synthetic Minority Over-sampling Technique) to identify fraudulent activity within large-scale transaction datasets. 

Key technical implementation details:
*   **Pipeline Architecture:** Uses `imblearn.pipeline` to ensure data transformations (including over-sampling) are only applied to the training set, preventing data leakage.
*   **Cyclical Feature Engineering:** A custom transformer converts raw time `step` values into cyclical `hour_sin` and `hour_cos` features, allowing the model to capture recurring daily fraud patterns.
*   **Production-Ready:** Encapsulated into a serialized `joblib` model for deployment.

## Data Design Decisions
* **Preventing Data Leakage:** I excluded `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, and `newbalanceDest`. In a real-time production setting, these balance updates are often reconciled *after* the transaction processing. Including them would provide the model with "future" knowledge (post-event information) that would not be available at the moment of prediction, leading to artificially inflated performance metrics that would not hold up in real-world deployment.
* **Overcoming Legacy Constraints:** I dropped `isFlaggedFraud` because it is derived from a simplistic, rule-based legacy system (a 200,000-unit threshold). By removing this, I forced the model to identify complex, non-linear patterns of fraud rather than simply learning to mimic a flawed, existing heuristic.

## Model Performance
The current iteration prioritizes **Recall** to ensure the highest percentage of fraudulent transactions are captured on the full dataset.

| Metric | Result |
| :--- | :--- |
| **Recall Score** | 0.6840 |
| **F1 Score** | 0.0435 |

## Usage
### Requirements
* `numpy`, `pandas`, `scikit-learn`, `imblearn`, `joblib`

### Execution
1. **Training:** Run the main pipeline script to train the model on the full dataset and save the `fraud_detection_pipeline.joblib` file.
2. **Prediction:** Use the following snippet in your production environment:

```python
import joblib
from src.custom_transformers import TimeFeatureTransformer

# Load the serialized pipeline
pipeline = joblib.load('fraud_detection_pipeline.joblib')

# Predict probabilities and apply a custom threshold for sensitivity
y_probs = pipeline.predict_proba(X_input)[:, 1]
is_fraud = (y_probs > 0.3).astype(int)

## Model Limitations & Future Work

* **False Positives:** While Recall has improved to ~0.68, the model currently flags a significant number of legitimate transactions. Future iterations will involve a cost-benefit analysis to determine the optimal decision threshold for business-specific risk tolerance.
* **Feature Engineering:** Future updates will explore interaction features between transaction amount and account type to further distinguish between legitimate high-value transfers and fraud.