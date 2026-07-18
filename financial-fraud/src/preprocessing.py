from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class TimeFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Convert step to hours
        hour = X['step'] % 24
        # Apply sine/cosine transform
        X['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        X['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        return X.drop(columns=['step'])