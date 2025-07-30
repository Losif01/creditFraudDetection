# src/data/resampler.py

# Strategy pattern used here for different strategies of resampling the imbalanced raw data
from abc import ABC, abstractmethod
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import SMOTE
import pandas as pd

class ResamplingStrategy(ABC):
    @abstractmethod
    def resample(self, X, y):
        pass

class RandomUnderSampling(ResamplingStrategy):
    def __init__(self, random_state=42):
        self.sampler = RandomUnderSampler(random_state=random_state)

    def resample(self, X, y):
        return self.sampler.fit_resample(X, y)

class NearMissSampling(ResamplingStrategy):
    def __init__(self, version=3):
        self.sampler = NearMiss(version=version)

    def resample(self, X, y):
        return self.sampler.fit_resample(X, y)

class SMOTESampling(ResamplingStrategy):
    def __init__(self):
        self.sampler = SMOTE()

    def resample(self, X, y):
        return self.sampler.fit_resample(X, y)

class NoSampling(ResamplingStrategy):
    def resample(self, X, y):
        return X, y