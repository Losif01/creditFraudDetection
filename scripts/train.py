#!/usr/bin/env python3
"""
Main training script for credit card fraud detection.
This script replicates the notebook functionality using our organized modules.
"""

import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.data_loader import DataLoader
from src.models.model_factory import ModelFactory
from src.training.trainer import ModelTrainer
from src.training.evaluation import ModelEvaluator
from src.utils.logger import get_project_logger

def main():
    """Main training pipeline."""
    # Setup logging
    logger = get_project_logger(__name__)
    logger.info("Starting Credit Card Fraud Detection Training Pipeline")
    
    try:
        # =============================================================================
        # 1. DATA LOADING AND PREPROCESSING
        # =============================================================================
        logger.info("Step 1: Loading and preprocessing data")
        
        data_loader = DataLoader()
        X_train, X_test, y_train, y_test = data_loader.load_and_preprocess()
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        
        # =============================================================================
        # 2. MODEL TRAINING
        # =============================================================================
        logger.info("Step 2: Training models")
        
        trainer = ModelTrainer()
        
        # Define models to train
        model_types = [
            'logistic_regression',
            'xgboost', 
            'mlp',
            'random_forest',
            'svm',
            'decision_tree'
        ]
        
        # Train all models
        results = trainer.train_multiple_models(
            model_types, X_train, y_train, X_test, y_test
        )
        
        # =============================================================================
        # 3. CROSS-VALIDATION
        # =============================================================================
        logger.info("Step 3: Performing cross-validation")
        
        # Combine balanced data for CV
        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([y_train, y_test])
        
        cv_results = trainer.cross_validate_models(model_types, X_full, y_full)
        
        # =============================================================================
        # 4. MODEL EVALUATION AND COMPARISON
        # =============================================================================
        logger.info("Step 4: Evaluating and comparing models")
        
        evaluator = ModelEvaluator(trainer.config)
        comparison_df = evaluator.compare_models(results)
        
        print("\n" + "="*60)
        print("FINAL MODEL COMPARISON RESULTS")
        print("="*60)
        print(comparison_df)
        
        # Get best model
        best_model_name, best_model = trainer.get_best_model('f1_weighted')
        best_score = results[best_model_name]['f1_weighted']
        
        print(f"\nBest Model: {best_model_name}")
        print(f"F1-Weighted Score: {best_score:.4f}")
        
        # =============================================================================
        # 5. HYPERPARAMETER TUNING (OPTIONAL)
        # =============================================================================
        if trainer.config.get('hyperparameter_tuning', {}).get('enabled', False):
            logger.info("Step 5: Hyperparameter tuning")
            
            # Tune the best model
            tuned_model, tuning_results = trainer.hyperparameter_tuning(
                best_model_name, X_train, y_train
            )
            
            # Evaluate tuned model
            tuned_metrics = evaluator.evaluate_single_model(tuned_model, X_test, y_test)
            
            print(f"\nTuned {best_model_name} Results:")
            print(f"Original F1 Score: {best_score:.4f}")
            print(f"Tuned F1 Score: {tuned_metrics['f1_weighted']:.4f}")
            print(f"Improvement: {tuned_metrics['f1_weighted'] - best_score:.4f}")
        
        # =============================================================================
        # 6. SAVE RESULTS
        # =============================================================================
        logger.info("Step 6: Saving results")
        
        # Save processed data
        data_loader.save_processed_data(X_train, X_test, y_train, y_test)
        
        # Save trained models
        trainer.save_trained_models()
        
        # Save evaluation results
        os.makedirs('results', exist_ok=True)
        evaluator.save_evaluation_results(results, 'results/model_evaluation_results.json')
        
        # Save comparison DataFrame
        comparison_df.to_csv('results/model_comparison.csv', index=True)
        
        logger.info("Training pipeline completed successfully!")
        
        return results, best_model_name, best_score
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train credit card fraud detection models'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=['logistic_regression', 'xgboost', 'mlp', 'random_forest'],
        help='List of models to train'
    )
    
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Enable hyperparameter tuning'
    )
    
    parser.add_argument(
        '--cv',
        action='store_true',
        help='Perform cross-validation'
    )
    
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save evaluation plots'
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Update matplotlib backend for headless environments
    if args.save_plots:
        plt.switch_backend('Agg')
    
    try:
        results, best_model, best_score = main()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Best Model: {best_model}")
        print(f"Best Score: {best_score:.4f}")
        print("\nCheck the following locations for results:")
        print("- models/saved/ (trained models)")
        print("- results/ (evaluation results)")
        print("- logs/ (training logs)")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        sys.exit(1)