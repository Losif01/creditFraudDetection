"""
Logistic Regression model for credit card fraud detection.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class LogisticRegressionModel(BaseModel):
    """Logistic Regression model implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Logistic Regression model."""
        super().__init__(config)
        self.model_config = config.get('models', {}).get('logistic_regression', {})
        self.model = self.build_model()
    
    def build_model(self) -> LogisticRegression:
        """Build and return LogisticRegression model."""
        params = {
            'C': self.model_config.get('C', 0.01),
            'penalty': self.model_config.get('penalty', 'l2'),
            'solver': self.model_config.get('solver', 'lbfgs'),
            'random_state': self.model_config.get('random_state', 42)
        }
        
        logger.info(f"Building Logistic Regression with params: {params}")
        return LogisticRegression(**params)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'LogisticRegressionModel':
        """Train the Logistic Regression model."""
        logger.info("Training Logistic Regression model...")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Log training score
        train_score = self.model.score(X_train, y_train)
        logger.info(f"Training accuracy: {train_score:.4f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def get_coefficients(self) -> np.ndarray:
        """Get model coefficients."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get coefficients")
        
        return self.model.coef_[0]
    
    def get_intercept(self) -> float:
        """Get model intercept."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get intercept")
        
        return self.model.intercept_[0]