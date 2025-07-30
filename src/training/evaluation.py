"""
Model evaluation utilities for credit card fraud detection.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.model_selection import cross_val_score, KFold
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation utilities."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize evaluator with configuration."""
        self.config = config
        self.eval_config = config.get('evaluation', {})
        self.metrics_config = self.eval_config.get('metrics', ['accuracy', 'f1_weighted'])
    
    def evaluate_single_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate a single model and return metrics."""
        logger.info(f"Evaluating {model.model_name}")
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {}
        
        if 'accuracy' in self.metrics_config:
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
        
        if 'f1_weighted' in self.metrics_config:
            metrics['f1_weighted'] = f1_score(y_test, y_pred, average='weighted')
        
        if 'f1_macro' in self.metrics_config:
            metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro')
        
        if 'precision' in self.metrics_config:
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
        
        if 'recall' in self.metrics_config:
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
        
        # ROC AUC if probabilities available
        if 'roc_auc' in self.metrics_config:
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
            except (NotImplementedError, AttributeError):
                logger.warning(f"ROC AUC not available for {model.model_name}")
        
        return metrics
    
    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Perform cross-validation on a model."""
        cv_config = self.config.get('training', {})
        cv_folds = cv_config.get('cv_folds', 5)
        cv_shuffle = cv_config.get('cv_shuffle', True)
        cv_random_state = cv_config.get('cv_random_state', 42)
        
        logger.info(f"Cross-validating {model.model_name} with {cv_folds} folds")
        
        kfold = KFold(n_splits=cv_folds, shuffle=cv_shuffle, random_state=cv_random_state)
        
        # Rebuild model for cross-validation
        cv_model = model.build_model()
        
        cv_scores = cross_val_score(cv_model, X, y, cv=kfold, scoring='accuracy')
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
    
    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray, 
                            model_name: str, figsize: Tuple[int, int] = (7, 5)):
        """Plot confusion matrix."""
        if not self.eval_config.get('plot_confusion_matrix', True):
            return None
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Fraud', 'Fraud'],
                   yticklabels=['Not Fraud', 'Fraud'])
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_roc_curve(self, models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series):
        """Plot ROC curves for multiple models."""
        if not self.eval_config.get('plot_roc_curve', True):
            return None
        
        plt.figure(figsize=(10, 8))
        
        for name, model in models.items():
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
            except (NotImplementedError, AttributeError):
                logger.warning(f"ROC curve not available for {name}")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        
        return plt.gcf()
    
    def generate_classification_report(self, y_true: pd.Series, y_pred: np.ndarray, 
                                     model_name: str) -> str:
        """Generate detailed classification report."""
        report = classification_report(y_true, y_pred, 
                                     target_names=['Not Fraud', 'Fraud'])
        
        logger.info(f"\n{model_name} Classification Report:\n{report}")
        return report
    
    def compare_models(self, model_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Compare multiple models and return results DataFrame."""
        comparison_df = pd.DataFrame(model_results).T
        comparison_df = comparison_df.round(4)
        
        # Sort by a primary metric (e.g., f1_weighted or accuracy)
        sort_metric = 'f1_weighted' if 'f1_weighted' in comparison_df.columns else 'accuracy'
        comparison_df = comparison_df.sort_values(by=sort_metric, ascending=False)
        
        logger.info(f"\nModel Comparison Results:\n{comparison_df}")
        return comparison_df
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, metric: str = 'accuracy'):
        """Plot model comparison for a specific metric."""
        if metric not in comparison_df.columns:
            logger.warning(f"Metric '{metric}' not found in comparison results")
            return None
        
        plt.figure(figsize=(12, 6))
        comparison_df[metric].plot(kind='bar')
        plt.title(f'Model Comparison - {metric.upper()}')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.xlabel('Models')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Add value labels on bars
        for i, v in enumerate(comparison_df[metric]):
            plt.text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom')
        
        return plt.gcf()
    
    def evaluate_feature_importance(self, model, feature_names: List[str], 
                                  top_n: int = 15) -> pd.DataFrame:
        """Evaluate and return feature importance."""
        try:
            importance = model.get_feature_importance()
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Get top N features
            top_features = importance_df.head(top_n)
            
            logger.info(f"\nTop {top_n} Important Features for {model.model_name}:")
            logger.info(f"\n{top_features}")
            
            return top_features
        
        except NotImplementedError:
            logger.warning(f"Feature importance not available for {model.model_name}")
            return pd.DataFrame()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, model_name: str):
        """Plot feature importance."""
        if importance_df.empty:
            return None
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance')
        plt.title(f'{model_name} - Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        return plt.gcf()
    
    def save_evaluation_results(self, results: Dict[str, Any], filepath: str):
        """Save evaluation results to file."""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v 
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value.tolist() if isinstance(value, np.ndarray) else value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {filepath}")