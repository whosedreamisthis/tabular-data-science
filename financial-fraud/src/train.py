import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from pipeline import get_fraud_pipeline

def train_and_save_model(data_path, output_path):
    # Load dataset
    df = pd.read_csv(data_path)

    # Feature selection
    df = df.drop(['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 
                  'isFlaggedFraud', 'nameOrig', 'nameDest'], axis=1)

    X = df.drop(["isFraud"], axis=1)
    y = df.isFraud

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get the pipeline from our module
    pipeline = get_fraud_pipeline()

    # Train
    pipeline.fit(X_train, y_train)

    # Persist
    joblib.dump(pipeline, output_path)
    
    print("Training completed.")

if __name__ == "__main__":
    train_and_save_model("../data/Synthetic_Financial_datasets_log.csv", "../models/fraud_model.joblib")