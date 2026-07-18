import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, StandardScaler
from preprocessing import * # Import all from your file
from sklearn.model_selection import GridSearchCV

# Load and Filter Data
df = pd.read_csv("../data/used_cars.csv")
df['price'] = df['price'].astype(str).str.replace(r'[$,]', '', regex=True).astype(float)

# CRITICAL FIX: Filter outliers to focus on the 'normal' market
df = df[(df['price'] >= 2000) & (df['price'] <= 150000)]

X = df.drop(columns=['price'])
y = np.log1p(df['price']) # Log-transform the target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipelines
mileage_pipeline = Pipeline([
    ('cleaner', NumericalStringCleaner(variables=['milage'])),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

flag_cols = ['accident', 'clean_title'] 

# Re-define your flag_pipeline using the updated list
flag_pipeline = Pipeline([
    ('standardizer', CategoricalStandardizer(variables=flag_cols)),
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

year_pipeline = Pipeline([
    ('cleaner', NumericalStringCleaner(variables=['model_year'])),
    ('age_calc', AgeCalculator(current_year=2026)),
    ('imputer', SimpleImputer(strategy='median'))
])

brand_model_pipeline = Pipeline([
    ('std', CategoricalStandardizer(variables=['brand', 'model'])),
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('grouper', RareCategoryGrouper(variables=['brand', 'model'], threshold=1)),
    ('encoder', TargetEncoder())
])

transmission_pipeline = Pipeline([
    ('standardizer', TransmissionStandardizer(variables=['transmission'])),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Update your ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num_mileage', mileage_pipeline, ['milage']),
        ('num_year', year_pipeline, ['model_year']),
        ('brand_model', brand_model_pipeline, ['brand', 'model']),
        # Re-add these one at a time to check their impact
        ('cat_transmission', transmission_pipeline, ['transmission']), 
        ('flags', flag_pipeline, ['accident', 'clean_title']) # Only include these two key flags
    ],
    remainder='drop'
)

# Model
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=5, n_jobs=-1, subsample= 1.0))
])



full_pipeline.fit(X_train, y_train)

# Predict and convert back from Log
y_pred = np.expm1(full_pipeline.predict(X_test))
y_test_orig = np.expm1(y_test)

from sklearn.metrics import r2_score
print(f"Final R2 Score: {r2_score(y_test_orig, y_pred)}")