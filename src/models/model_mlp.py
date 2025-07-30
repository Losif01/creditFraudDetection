"""
Multi-Layer Perceptron (Neural Network) model for credit card fraud detection.
"""
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from .base_model import BaseModel
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class MLPModel(BaseModel):
    """Multi-Layer Perceptron model implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MLP model."""
        super().__init__(config)
        self.model_config = config.get('models', {}).get('mlp', {})
        self.model = self.build_model()
    
    def build_model(self) -> MLPClassifier:
        """Build and return MLPClassifier model."""
        params = {
            'hidden_layer_sizes': tuple(self.model_config.get('hidden_layer_sizes', [50, 50])),
            'max_iter': self.model_config.get('max_iter', 1000),
            'random_state': self.model_config.get('random_state', 42),
            'alpha': self.model_config.get('alpha', 0.0001),
            'learning_rate': self.model_config.get('learning_rate', 'constant'),
            'solver': self.model_config.get('solver', 'adam')
        }
        
        logger.info(f"Building MLP with params: {params}")
        return MLPClassifier(**params)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'MLPModel':
        """Train the MLP model."""
        logger.info("Training MLP model...")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Log training score and convergence info
        train_score = self.model.score(X_train, y_train)
        logger.info(f"Training accuracy: {train_score:.4f}")
        logger.info(f"Converged after {self.model.n_iter_} iterations")
        
        if not self.model.n_iter_ < self.model.max_iter:
            logger.warning("Model did not converge. Consider increasing max_iter.")
        
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
    
    def get_loss_curve(self) -> np.ndarray:
        """Get the loss curve during training."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get loss curve")
        
        return self.model.loss_curve_
    
    def plot_loss_curve(self):
        """Plot the training loss curve."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.is_trained:
                raise ValueError("Model must be trained to plot loss curve")
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.model.loss_curve_)
            plt.title('MLP Training Loss Curve')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.show()
        except ImportError:
            logger.warning("matplotlib not available for plotting")
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get information about the trained network."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get network info")
        
        return {
            'layers': self.model.hidden_layer_sizes,
            'n_layers': self.model.n_layers_,
            'n_iter': self.model.n_iter_,
            'loss': self.model.loss_,
            'output_activation': self.model.out_activation_
        }