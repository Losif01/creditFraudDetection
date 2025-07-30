# src/features/feature_engineering.py
from utils.helpers import ensure_dir_exists
from sklearn.preprocessing import RobustScaler
import pandas as pd

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.scaler = None

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.config["features"]["scale"]:
            cols = self.config["features"]["features_to_scale"]
            self.scaler = RobustScaler()
            X[cols] = self.scaler.fit_transform(X[cols])
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """For inference — use fitted scaler."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        cols = self.config["features"]["features_to_scale"]
        X[cols] = self.scaler.transform(X[cols])
        return X

    def save_scaler(self, path: str):
        import joblib
        ensure_dir_exists(path)
        joblib.dump(self.scaler, path)

    def load_scaler(self, path: str):
        import joblib
        self.scaler = joblib.load(path)