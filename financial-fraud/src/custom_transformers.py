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

	def get_feature_names_out(self, input_features=None):
        	# Return the names of the features your transformer outputs
	        # If your transformer just transforms the existing column:
        	return np.array(['hour_sin', 'hour_cos'])