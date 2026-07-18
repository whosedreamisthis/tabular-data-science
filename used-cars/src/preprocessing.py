import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CategoricalStandardizer(BaseEstimator, TransformerMixin):
    def __init__(self, variables, tracking_dirty_values=None):
        self.variables = variables
        self.tracking_dirty_values = tracking_dirty_values or ['–', 'not supported', 'none', 'null']
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        for col in self.variables:
            if col in X.columns:
                X[col] = X[col].astype(str).str.strip().replace(self.tracking_dirty_values + ['nan', 'NaN'], np.nan)
        return X

class NumericalStringCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, variables, tracking_dirty_values=None):
        self.variables = variables
        self.tracking_dirty_values = tracking_dirty_values or ['–', 'not supported', 'none', 'null']
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        for col in self.variables:
            if col in X.columns:
                X[col] = X[col].astype(str).str.replace(r'[$, mi,]', '', regex=True)
                X[col] = X[col].replace(self.tracking_dirty_values + ['nan', 'NaN'], np.nan)
                X[col] = pd.to_numeric(X[col], errors='coerce')
        return X

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, variables, threshold=5):
        self.variables = variables
        self.threshold = threshold
    def fit(self, X, y=None):
        self.frequent_ = {col: X[col].value_counts()[X[col].value_counts() >= self.threshold].index.tolist() for col in self.variables}
        return self
    def transform(self, X):
        X = X.copy()
        for col in self.variables:
            X[col] = X[col].apply(lambda v: v if v in self.frequent_[col] else 'Other')
        return X

class AgeCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, current_year=2026):
        self.current_year = current_year
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        # Create car_age without dropping model_year (ColumnTransformer handles dropping)
        X['car_age'] = self.current_year - pd.to_numeric(X['model_year'], errors='coerce')
        return X

# Keep your ZeroMileageFlagCreator, TransmissionStandardizer, and ColorStandardizer as they are.