# src/ml_pipeline.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

class GenericScikitLearner:
    """Uniwersalny wrapper dla scikit-learn classifierów"""
    
    def __init__(self, classifier, name="classifier"):
        self.classifier = classifier
        self.name = name
        self.scaler = StandardScaler()
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            (name, self.classifier)
        ])
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Trening"""
        X_processed = X.select_dtypes(include=[np.number])  # Tylko numeryczne
        self.pipeline.fit(X_processed, y)
        return self
    
    def predict(self, X: pd.DataFrame):
        """Predykcja"""
        X_processed = X.select_dtypes(include=[np.number])
        return self.pipeline.predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame):
        """Predykcja z prawdopodobieństwami"""
        X_processed = X.select_dtypes(include=[np.number])
        return self.pipeline.predict_proba(X_processed)
    
    def score(self, X: pd.DataFrame, y: pd.Series):
        """Ocena modelu"""
        X_processed = X.select_dtypes(include=[np.number])
        return self.pipeline.score(X_processed, y)
    
    def save(self, path: str):
        """Zapisz model"""
        joblib.dump(self.pipeline, path)
    
    def load(self, path: str):
        """Wczytaj model"""
        self.pipeline = joblib.load(path)
        return self
