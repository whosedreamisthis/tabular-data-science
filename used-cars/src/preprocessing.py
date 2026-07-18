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
        # If X is an array (which happens in ColumnTransformer), convert to DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.variables)
            
        self.frequent_ = {}
        for col in self.variables:
            # Safely count values
            counts = X[col].value_counts()
            self.frequent_[col] = counts[counts >= self.threshold].index.tolist()
        return self
        
    def transform(self, X):
        # Ensure X is a DataFrame for consistent indexing
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.variables)
            
        X = X.copy()
        for col in self.variables:
            # Map values, using .get() to handle unseen categories safely
            X[col] = X[col].apply(lambda v: v if v in self.frequent_.get(col, []) else 'Other')
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

class TransmissionStandardizer(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        for col in self.variables:
            if col in X.columns:
                lower_series = X[col].astype(str).str.lower().str.strip()
                # Standardize common variations to binary-like categories
                standardized = pd.Series("Unknown", index=X.index)
                is_manual = lower_series.str.contains(r'manual|m/t|mt|man\.', regex=True)
                standardized[is_manual] = "Manual"
                is_auto = lower_series.str.contains(r'automatic|a/t|at|auto|cvt|dual-clutch|dct', regex=True)
                standardized[is_auto] = "Automatic"
                X[col] = standardized
        return X