"""
Data loading and preprocessing module for credit card fraud detection.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import yaml
import os
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles data loading and preprocessing operations."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize DataLoader with configuration."""
        # Resolve config path relative to this file if needed
        config_abs_path = os.path.abspath(config_path)
        if not os.path.exists(config_abs_path):
            raise FileNotFoundError(f"Config file not found: {config_abs_path}")

        with open(config_abs_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Base directory of the config file (e.g., ./config)
        config_dir = os.path.dirname(config_abs_path)

        # Resolve data paths relative to config file location
        raw_path = self.config['data']['raw_path']
        self.config['data']['raw_path'] = os.path.join(config_dir, raw_path)

        processed_path = self.config['data']['processed_path']
        self.config['data']['processed_path'] = os.path.join(config_dir, processed_path)

        # Optional: normalize path (handles ../ correctly)
        self.config['data']['raw_path'] = os.path.normpath(self.config['data']['raw_path'])
        self.config['data']['processed_path'] = os.path.normpath(self.config['data']['processed_path'])

        self.scaler = None
        self._setup_scaler()
    
    def _setup_scaler(self):
        """Setup the appropriate scaler based on configuration."""
        scaler_type = self.config['preprocessing']['scaling']['method']
        
        if scaler_type == "robust":
            self.scaler = RobustScaler()
        elif scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw credit card data."""
        raw_path = self.config['data']['raw_path']
        
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw data file not found: {raw_path}")
        
        logger.info(f"Loading data from {raw_path}")
        df = pd.read_csv(raw_path)
        
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        logger.info(f"Fraud cases: {df['Class'].sum()}")
        logger.info(f"Fraud percentage: {(df['Class'].sum() / len(df)) * 100:.3f}%")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset by removing duplicates and handling missing values."""
        logger.info("Starting data cleaning...")
        
        # Check for missing values
        missing_values = df.isna().sum().sum()
        if missing_values > 0:
            logger.warning(f"Found {missing_values} missing values")
        else:
            logger.info("No missing values found")
        
        # Remove duplicates if configured
        if self.config['preprocessing']['remove_duplicates']:
            initial_shape = df.shape
            duplicates = df.duplicated().sum()
            df = df.drop_duplicates()
            
            logger.info(f"Removed {duplicates} duplicates")
            logger.info(f"Shape changed from {initial_shape} to {df.shape}")
        
        return df
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale specified features."""
        features_to_scale = self.config['preprocessing']['scaling']['features']
        
        if features_to_scale:
            logger.info(f"Scaling features: {features_to_scale}")
            df_scaled = df.copy()
            df_scaled[features_to_scale] = self.scaler.fit_transform(df[features_to_scale])
            return df_scaled
        
        return df
    
    def balance_dataset(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Balance the dataset using the specified sampling method."""
        sampling_method = self.config['preprocessing']['sampling']['method']
        random_state = self.config['preprocessing']['sampling']['random_state']
        
        logger.info(f"Balancing dataset using {sampling_method}")
        logger.info(f"Original class distribution: {y.value_counts().to_dict()}")
        
        if sampling_method == "undersample":
            # Manual undersampling as shown in notebook
            fraud = X[y == 1]
            not_fraud = X[y == 0].sample(n=len(fraud), random_state=random_state)
            
            X_balanced = pd.concat([fraud, not_fraud])
            y_balanced = pd.concat([y[y == 1], y[y == 0].sample(n=len(fraud), random_state=random_state)])
            
            # Shuffle the dataset
            combined = pd.concat([X_balanced, y_balanced], axis=1)
            combined = combined.sample(frac=1, random_state=random_state)
            
            X_balanced = combined.drop(columns=[self.config['features']['target_column']])
            y_balanced = combined[self.config['features']['target_column']]
            
        elif sampling_method == "oversample":
            sampler = SMOTE(random_state=random_state)
            X_balanced, y_balanced = sampler.fit_resample(X, y)
            X_balanced = pd.DataFrame(X_balanced, columns=X.columns)
            y_balanced = pd.Series(y_balanced)
            
        else:
            logger.info("No sampling applied")
            return X, y
        
        logger.info(f"Balanced class distribution: {y_balanced.value_counts().to_dict()}")
        return X_balanced, y_balanced
    
    def prepare_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and target variable."""
        target_col = self.config['features']['target_column']
        drop_cols = self.config['features']['drop_columns']
        
        # Prepare features
        X = df.drop(columns=[target_col] + drop_cols)
        y = df[target_col]
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        test_size = self.config['data']['test_size']
        random_state = self.config['data']['random_state']
        stratify = y if self.config['data']['stratify'] else None
        
        logger.info(f"Splitting data with test_size={test_size}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            stratify=stratify, 
            random_state=random_state
        )
        
        logger.info(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def load_and_preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Complete data loading and preprocessing pipeline."""
        logger.info("Starting complete data preprocessing pipeline...")
        
        # Load and clean data
        df = self.load_raw_data()
        df = self.clean_data(df)
        
        # Scale features
        df = self.scale_features(df)
        
        # Prepare features and target
        X, y = self.prepare_features_target(df)
        
        # Balance dataset
        X, y = self.balance_dataset(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        logger.info("Data preprocessing completed successfully!")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                           y_train: pd.Series, y_test: pd.Series):
        """Save processed data to files."""
        processed_path = self.config['data']['processed_path']
        os.makedirs(processed_path, exist_ok=True)
        
        X_train.to_csv(os.path.join(processed_path, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(processed_path, 'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(processed_path, 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(processed_path, 'y_test.csv'), index=False)
        
        logger.info(f"Processed data saved to {processed_path}")

if __name__ == "__main__":
    # Example usage
    data_loader = DataLoader()
    X_train, X_test, y_train, y_test = data_loader.load_and_preprocess()
    data_loader.save_processed_data(X_train, X_test, y_train, y_test)