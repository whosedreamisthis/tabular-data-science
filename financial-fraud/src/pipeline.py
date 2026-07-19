from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from custom_transformers import TimeFeatureTransformer

def get_fraud_pipeline():
    # Define preprocessing
    preprocessor = ColumnTransformer(transformers=[
        ('ohe', OneHotEncoder(), ["type"]),
        ('step', TimeFeatureTransformer(), ["step"])
    ], remainder="passthrough")

    # Construct the pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(sampling_strategy=0.1, random_state=42)),
        ('classifier', RandomForestClassifier(
            n_estimators=100, 
            min_samples_leaf=2,
            max_depth=10, 
            min_samples_split=5,
            class_weight='balanced_subsample',
            random_state=42
        ))
    ])
    
    return pipeline