import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define your business costs
COST_PER_FALSE_NEGATIVE = 1000  # Cost of missed fraud
COST_PER_FALSE_POSITIVE = 5     # Cost of manual review

def calculate_total_cost(y_true, y_probs, threshold):
    y_pred = (y_probs >= threshold).astype(int)
    
    # Calculate confusion matrix components
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    # Total cost formula
    total_cost = (fn * COST_PER_FALSE_NEGATIVE) + (fp * COST_PER_FALSE_POSITIVE)
    return total_cost

# Load the saved pipeline
pipeline = joblib.load('../model/fraud_detection_pipeline.joblib')

# Assume you have loaded your test data (e.g., X_test, y_test)

df = pd.read_csv("../data/Synthetic_Financial_datasets_log.csv")

# Remove redundant or identifying columns that do not contribute to fraud detection
df = df.drop(['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 
              'isFlaggedFraud', 'nameOrig', 'nameDest'], axis=1)

# Separate features and target variable
X = df.drop(["isFraud"], axis=1)
y = df.isFraud

# Split data into training and testing sets to evaluate model generalization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Get probabilities from your pipeline
y_probs = pipeline.predict_proba(X_test)[:, 1]

# Test thresholds from 0.01 to 0.99
thresholds = np.linspace(0.01, 0.99, 100)
costs = [calculate_total_cost(y_test, y_probs, t) for t in thresholds]

# Find the threshold with the minimum cost
best_threshold = thresholds[np.argmin(costs)]
min_cost = min(costs)

print(f"Optimal Threshold: {best_threshold:.4f}")
print(f"Minimum Total Cost: ${min_cost:,.2f}")

def plot_cost_curve(y_test, y_probs, thresholds, costs, best_threshold):
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, costs, label='Total Operational Cost', color='blue', linewidth=2)
    
    # Add a vertical line at the optimal threshold
    plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Optimal Threshold: {best_threshold:.4f}')
    
    plt.title('Cost-Benefit Analysis: Minimizing Operational Risk', fontsize=14)
    plt.xlabel('Probability Threshold', fontsize=12)
    plt.ylabel('Total Cost ($)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot for your README
    plt.savefig('cost_benefit_analysis.png')
    plt.show()

# Use the variables from your optimization script
plot_cost_curve(y_test, y_probs, thresholds, costs, best_threshold)