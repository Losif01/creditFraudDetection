"""
XGBoost model for credit card fraud detection.
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from .base_model import BaseModel
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class XGBoostModel(BaseModel):
    """XGBoost model implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize XGBoost model."""
        super().__init__(config)
        self.model_config = config.get('models', {}).get('xgboost', {})
        self.model = self.build_model()
    
    def build_model(self) -> XGBClassifier:
        """Build and return XGBClassifier model."""
        params = {
            'random_state': self.model_config.get('random_state', 42),
            'n_estimators': self.model_config.get('n_estimators', 100),
            'max_depth': self.model_config.get('max_depth', 6),
            'learning_rate': self.model_config.get('learning_rate', 0.1),
            'subsample': self.model_config.get('subsample', 1.0),
            'colsample_bytree': self.model_config.get('colsample_bytree', 1.0),
            'eval_metric': 'logloss'  # Suppress warning
        }
        
        logger.info(f"Building XGBoost with params: {params}")
        return XGBClassifier(**params)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'XGBoostModel':
        """Train the XGBoost model."""
        logger.info("Training XGBoost model...")
        
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
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        return self.model.feature_importances_
    
    def plot_importance(self, max_num_features: int = 20):
        """Plot feature importance."""
        try:
            from xgboost import plot_importance
            import matplotlib.pyplot as plt
            
            if not self.is_trained:
                raise ValueError("Model must be trained to plot importance")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            plot_importance(self.model, ax=ax, max_num_features=max_num_features)
            plt.title("XGBoost Feature Importance")
            plt.tight_layout()
            return fig
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return None