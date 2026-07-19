import pandas as pd
import joblib
from sklearn.metrics import f1_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
# Import your custom class so joblib can reconstruct the pipeline
from src.custom_transformers import TimeFeatureTransformer 

# 1. Load the saved pipeline
pipeline = joblib.load('../model/fraud_detection_pipeline.joblib')

# 2. Assume you have loaded your test data (e.g., X_test, y_test)

df = pd.read_csv("../data/Synthetic_Financial_datasets_log.csv")

# Remove redundant or identifying columns that do not contribute to fraud detection
df = df.drop(['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 
              'isFlaggedFraud', 'nameOrig', 'nameDest'], axis=1)

# Separate features and target variable
X = df.drop(["isFraud"], axis=1)
y = df.isFraud

# Split data into training and testing sets to evaluate model generalization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 3. Use predict_proba instead of predict to apply your custom threshold
y_probs = pipeline.predict_proba(X_test)[:, 1]
y_pred_custom = (y_probs > 0.3).astype(int)

# 4. Print metrics
print(f"F1 Score: {f1_score(y_test, y_pred_custom):.4f}")
print(f"Recall Score: {recall_score(y_test, y_pred_custom):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_custom))

# 5. Extract and print feature importance
rf = pipeline.named_steps['classifier']
preprocessor = pipeline.named_steps['preprocessor']
feature_names = preprocessor.get_feature_names_out()

importances = pd.Series(rf.feature_importances_, index=feature_names)
print("\nTop 10 Feature Importances:")
print(importances.sort_values(ascending=False).head(10))