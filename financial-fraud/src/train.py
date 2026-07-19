import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.pipeline import Pipeline 
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from src.custom_transformers import TimeFeatureTransformer
from sklearn.metrics import f1_score, confusion_matrix, recall_score
import joblib

# Load the full financial transaction dataset
df = pd.read_csv("../data/Synthetic_Financial_datasets_log.csv")

# Remove redundant or identifying columns that do not contribute to fraud detection
df = df.drop(['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 
              'isFlaggedFraud', 'nameOrig', 'nameDest'], axis=1)

# Separate features and target variable
X = df.drop(["isFraud"], axis=1)
y = df.isFraud

# Split data into training and testing sets to evaluate model generalization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing: Encode categorical 'type' and transform 'step' into cyclical time features
preprocessor = ColumnTransformer(transformers=[
    ('ohe', OneHotEncoder(), ["type"]),
    ('step', TimeFeatureTransformer(), ["step"])
], remainder="passthrough")

# Construct the pipeline: Preprocess -> Apply SMOTE for class balancing -> Train Random Forest
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(sampling_strategy=0.1, random_state=42)),
    ('classifier', RandomForestClassifier(
        n_estimators=100, 
        min_samples_leaf=2,
        max_depth=10, 
        min_samples_split=5,
        class_weight='balanced_subsample', # Aggressive balancing to handle minority fraud class
        random_state=42
    ))
])

# Train the finalized model on the full training dataset
pipeline.fit(X_train, y_train)

# Persist the trained pipeline to disk for production use
joblib.dump(pipeline, '../model/fraud_detection_pipeline.joblib')