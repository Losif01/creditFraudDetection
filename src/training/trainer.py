"""
Model training orchestrator for credit card fraud detection.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import GridSearchCV
import logging
import os
import yaml

from ..models.model_factory import ModelFactory
from ..models.base_model import BaseModel
from .evaluation import ModelEvaluator

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles training, hyperparameter tuning, and evaluation of models."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize trainer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.evaluator = ModelEvaluator(self.config)
        self.trained_models = {}
        self.evaluation_results = {}
    
    def train_single_model(self, model_type: str, X_train: pd.DataFrame, 
                          y_train: pd.Series, X_test: pd.DataFrame, 
                          y_test: pd.Series) -> Tuple[BaseModel, Dict[str, float]]:
        """Train a single model and evaluate it."""
        logger.info(f"Training {model_type} model...")
        
        # Create model
        model = ModelFactory.create_model(model_type, self.config)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        metrics = self.evaluator.evaluate_single_model(model, X_test, y_test)
        
        # Store results
        self.trained_models[model_type] = model
        self.evaluation_results[model_type] = metrics
        
        logger.info(f"{model_type} training completed. Accuracy: {metrics.get('accuracy', 0):.4f}")
        
        return model, metrics
    
    def train_multiple_models(self, model_types: List[str], X_train: pd.DataFrame, 
                            y_train: pd.Series, X_test: pd.DataFrame, 
                            y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """Train multiple models and compare them."""
        logger.info(f"Training {len(model_types)} models: {model_types}")
        
        all_results = {}
        
        for model_type in model_types:
            try:
                model, metrics = self.train_single_model(
                    model_type, X_train, y_train, X_test, y_test
                )
                all_results[model_type] = metrics
                
            except Exception as e:
                logger.error(f"Failed to train {model_type}: {str(e)}")
                continue
        
        # Generate comparison
        comparison_df = self.evaluator.compare_models(all_results)
        
        return all_results
    
    def cross_validate_models(self, model_types: List[str], X: pd.DataFrame, 
                            y: pd.Series) -> Dict[str, Dict[str, float]]:
        """Perform cross-validation on multiple models."""
        logger.info(f"Cross-validating {len(model_types)} models")
        
        cv_results = {}
        
        for model_type in model_types:
            try:
                model = ModelFactory.create_model(model_type, self.config)
                cv_metrics = self.evaluator.cross_validate_model(model, X, y)
                cv_results[model_type] = cv_metrics
                
                logger.info(f"{model_type} CV Score: {cv_metrics['cv_mean']:.4f} "
                          f"(±{cv_metrics['cv_std']:.4f})")
                
            except Exception as e:
                logger.error(f"Failed to cross-validate {model_type}: {str(e)}")
                continue
        
        return cv_results
    
    def hyperparameter_tuning(self, model_type: str, X_train: pd.DataFrame, 
                            y_train: pd.Series) -> Tuple[BaseModel, Dict[str, Any]]:
        """Perform hyperparameter tuning for a specific model."""
        if not self.config.get('hyperparameter_tuning', {}).get('enabled', False):
            logger.info("Hyperparameter tuning is disabled")
            return self.train_single_model(model_type, X_train, y_train, X_train, y_train)[0], {}
        
        logger.info(f"Starting hyperparameter tuning for {model_type}")
        
        # Get hyperparameter grid from config
        tuning_config = self.config.get('hyperparameter_tuning', {})
        param_grid = tuning_config.get(model_type, {})
        
        if not param_grid:
            logger.warning(f"No hyperparameter grid found for {model_type}")
            return self.train_single_model(model_type, X_train, y_train, X_train, y_train)[0], {}
        
        # Create base model
        model = ModelFactory.create_model(model_type, self.config)
        
        # Setup GridSearchCV
        cv_folds = tuning_config.get('cv_folds', 5)
        grid_search = GridSearchCV(
            model.model, 
            param_grid, 
            cv=cv_folds, 
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Create optimized model
        optimized_model = ModelFactory.create_model(model_type, self.config)
        optimized_model.model.set_params(**grid_search.best_params_)
        optimized_model.fit(X_train, y_train)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Best parameters for {model_type}: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return optimized_model, results
    
    def train_with_kfold_comparison(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Train models using K-fold cross-validation as shown in notebook."""
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
        from xgboost import XGBClassifier
        
        # Define classifiers as in the notebook
        classifiers = {
            "LogisticRegression": LogisticRegression(),
            "KNearest": KNeighborsClassifier(),
            "Support Vector Classifier": SVC(),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            'XGBClassifier': XGBClassifier()
        }
        
        results = []
        
        for name, classifier in classifiers.items():
            try:
                cv_results = self.evaluator.cross_validate_model(
                    type('temp', (), {'model': classifier, 'model_name': name, 'build_model': lambda: classifier})(),
                    X, y
                )
                
                results.append({
                    'model': name,
                    'cv_score': cv_results['cv_mean'],
                    'cv_std': cv_results['cv_std']
                })
                
                logger.info(f"{name}: {cv_results['cv_mean']:.4f} (±{cv_results['cv_std']:.4f})")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('cv_score', ascending=False)
        
        return results_df
    
    def train_ensemble_models(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Train ensemble models as shown in the notebook."""
        from sklearn.ensemble import (
            AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier,
            RandomForestClassifier
        )
        from sklearn.tree import ExtraTreeClassifier
        from sklearn.model_selection import cross_val_score
        
        models = [
            AdaBoostClassifier(),
            BaggingClassifier(),
            GradientBoostingClassifier(),
            ExtraTreeClassifier(),
            RandomForestClassifier()
        ]
        
        results = []
        
        for model in models:
            try:
                cv_scores = cross_val_score(model, X, y, scoring='accuracy', cv=5)
                results.append({
                    'model': model.__class__.__name__,
                    'cv_score': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                })
                
                logger.info(f"{model.__class__.__name__}: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                
            except Exception as e:
                logger.error(f"Error training {model.__class__.__name__}: {str(e)}")
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('cv_score', ascending=False)
        
        return results_df
    
    def perform_grid_search_comparison(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        """Perform grid search as shown in the notebook."""
        from sklearn.naive_bayes import GaussianNB
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        
        model_params = {
            'logistic_regression': {
                'model': LogisticRegression(),
                'parameter': {
                    'solver': ['liblinear']
                }
            },
            'svm': {
                'model': SVC(),
                'parameter': {
                    'kernel': ['rbf', 'linear'],
                    'C': [10, 15, 20]
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(),
                'parameter': {
                    'criterion': ['gini', 'entropy']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(),
                'parameter': {
                    'criterion': ['gini', 'entropy'],
                    'n_estimators': [50, 100, 150]
                }
            },
            'naive_bayes_gaussian': {
                'model': GaussianNB(),
                'parameter': {}
            },
            'k_nearest_neighbors': {
                'model': KNeighborsClassifier(),
                'parameter': {
                    'n_neighbors': [5, 10, 15]
                }
            }
        }
        
        results = []
        
        for model_name, mp in model_params.items():
            try:
                clf = GridSearchCV(mp['model'], mp['parameter'], cv=5)
                clf.fit(X_train, y_train)
                
                results.append({
                    'model': model_name,
                    'best_score': clf.best_score_,
                    'best_params': clf.best_params_
                })
                
                logger.info(f"{model_name}: {clf.best_score_:.4f} - {clf.best_params_}")
                
            except Exception as e:
                logger.error(f"Error with grid search for {model_name}: {str(e)}")
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('best_score', ascending=False)
        
        return results_df
    
    def save_trained_models(self, save_dir: str = "models/saved"):
        """Save all trained models."""
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            model_path = os.path.join(save_dir, f"{model_name}.pkl")
            model.save_model(model_path)
            logger.info(f"Saved {model_name} to {model_path}")
    
    def load_trained_models(self, load_dir: str = "models/saved"):
        """Load previously trained models."""
        if not os.path.exists(load_dir):
            logger.warning(f"Load directory {load_dir} does not exist")
            return
        
        for filename in os.listdir(load_dir):
            if filename.endswith('.pkl'):
                model_name = filename[:-4]  # Remove .pkl extension
                model_path = os.path.join(load_dir, filename)
                
                # Create model instance and load
                try:
                    model = ModelFactory.create_model(model_name, self.config)
                    model.load_model(model_path)
                    self.trained_models[model_name] = model
                    logger.info(f"Loaded {model_name} from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {str(e)}")
    
    def get_best_model(self, metric: str = 'f1_weighted') -> Tuple[str, BaseModel]:
        """Get the best performing model based on a metric."""
        if not self.evaluation_results:
            raise ValueError("No models have been evaluated yet")
        
        best_model_name = max(
            self.evaluation_results.keys(),
            key=lambda x: self.evaluation_results[x].get(metric, 0)
        )
        
        best_model = self.trained_models[best_model_name]
        best_score = self.evaluation_results[best_model_name][metric]
        
        logger.info(f"Best model: {best_model_name} with {metric}: {best_score:.4f}")
        
        return best_model_name, best_model