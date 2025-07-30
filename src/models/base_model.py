"""
Abstract base class for all machine learning models.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import logging

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for all machine learning models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize base model with configuration."""
        self.config = config
        self.model = None
        self.is_trained = False
        self.model_name = self.__class__.__name__
    
    @abstractmethod
    def build_model(self) -> Any:
        """Build and return the model instance."""
        pass
    
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'BaseModel':
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities if supported."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError(f"{self.model_name} does not support probability predictions")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted')
        }
        
        # Add ROC AUC if probability predictions are available
        try:
            y_proba = self.predict_proba(X_test)[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        except (NotImplementedError, AttributeError):
            logger.warning(f"{self.model_name} does not support ROC AUC calculation")
        
        return metrics
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance if supported."""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0])
        else:
            raise NotImplementedError(f"{self.model_name} does not support feature importance")
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        import pickle
        
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        import pickle
        
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        if self.model is not None:
            return self.model.get_params()
        return {}
    
    def set_params(self, **params):
        """Set model parameters."""
        if self.model is not None:
            self.model.set_params(**params)
        else:
            raise ValueError("Model must be built before setting parameters")
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.model_name}(trained={self.is_trained})"
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return self.__str__()