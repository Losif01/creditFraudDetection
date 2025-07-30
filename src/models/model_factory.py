"""
Factory pattern for creating machine learning models.
"""
from typing import Dict, Any
from .base_model import BaseModel
from .model_logistic import LogisticRegressionModel
from .model_xgb import XGBoostModel
from .model_mlp import MLPModel
import logging

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory class for creating machine learning models."""
    
    _models = {
        'logistic_regression': LogisticRegressionModel,
        'logistic': LogisticRegressionModel,
        'lr': LogisticRegressionModel,
        'xgboost': XGBoostModel,
        'xgb': XGBoostModel,
        'mlp': MLPModel,
        'neural_network': MLPModel,
        'nn': MLPModel
    }
    
    @classmethod
    def create_model(cls, model_type: str, config: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance based on the model type.
        
        Args:
            model_type: Type of model to create ('logistic_regression', 'xgboost', 'mlp', etc.)
            config: Configuration dictionary
            
        Returns:
            BaseModel: Instance of the requested model
            
        Raises:
            ValueError: If model_type is not supported
        """
        model_type_lower = model_type.lower()
        
        if model_type_lower not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Available models: {available_models}")
        
        model_class = cls._models[model_type_lower]
        logger.info(f"Creating {model_class.__name__} model")
        
        return model_class(config)
    
    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available model types."""
        return list(set(cls._models.values()))
    
    @classmethod
    def register_model(cls, model_name: str, model_class: type):
        """
        Register a new model type.
        
        Args:
            model_name: Name to register the model under
            model_class: Model class that inherits from BaseModel
        """
        if not issubclass(model_class, BaseModel):
            raise ValueError("Model class must inherit from BaseModel")
        
        cls._models[model_name.lower()] = model_class
        logger.info(f"Registered new model: {model_name} -> {model_class.__name__}")

# Example of how to add sklearn models dynamically
def create_sklearn_model_wrapper(sklearn_model_class, default_params=None):
    """Create a wrapper for sklearn models to work with our BaseModel interface."""
    
    class SklearnModelWrapper(BaseModel):
        def __init__(self, config: Dict[str, Any]):
            super().__init__(config)
            self.sklearn_class = sklearn_model_class
            self.default_params = default_params or {}
            self.model = self.build_model()
        
        def build_model(self):
            # Get model-specific config or use defaults
            model_name = self.sklearn_class.__name__.lower()
            model_config = self.config.get('models', {}).get(model_name, {})
            
            # Merge with defaults
            params = {**self.default_params, **model_config}
            
            logger.info(f"Building {self.sklearn_class.__name__} with params: {params}")
            return self.sklearn_class(**params)
        
        def fit(self, X_train, y_train):
            logger.info(f"Training {self.sklearn_class.__name__} model...")
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            # Log training score if available
            if hasattr(self.model, 'score'):
                train_score = self.model.score(X_train, y_train)
                logger.info(f"Training accuracy: {train_score:.4f}")
            
            return self
        
        def predict(self, X):
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            return self.model.predict(X)
    
    return SklearnModelWrapper

# Register additional sklearn models
try:
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    
    # Register additional models
    ModelFactory.register_model('random_forest', create_sklearn_model_wrapper(
        RandomForestClassifier, {'random_state': 42}))
    ModelFactory.register_model('rf', create_sklearn_model_wrapper(
        RandomForestClassifier, {'random_state': 42}))
    ModelFactory.register_model('svm', create_sklearn_model_wrapper(SVC, {'probability': True}))
    ModelFactory.register_model('decision_tree', create_sklearn_model_wrapper(
        DecisionTreeClassifier, {'random_state': 42}))
    ModelFactory.register_model('dt', create_sklearn_model_wrapper(
        DecisionTreeClassifier, {'random_state': 42}))
    ModelFactory.register_model('knn', create_sklearn_model_wrapper(KNeighborsClassifier))
    ModelFactory.register_model('naive_bayes', create_sklearn_model_wrapper(GaussianNB))
    ModelFactory.register_model('nb', create_sklearn_model_wrapper(GaussianNB))
    ModelFactory.register_model('adaboost', create_sklearn_model_wrapper(
        AdaBoostClassifier, {'random_state': 42}))
    ModelFactory.register_model('gradient_boosting', create_sklearn_model_wrapper(
        GradientBoostingClassifier, {'random_state': 42}))
    ModelFactory.register_model('gb', create_sklearn_model_wrapper(
        GradientBoostingClassifier, {'random_state': 42}))
    
except ImportError as e:
    logger.warning(f"Some sklearn models not available: {e}")