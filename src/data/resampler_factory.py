# src/data/resampler_factory.py
# using factory pattern to use the strategy pattern in resampler.py
from resampler import RandomUnderSampling, NearMissSampling, SMOTESampling, NoSampling
class ResamplerFactory:
    @staticmethod
    def create(config):
        method = config["sampling"]["method"]
        random_state = config["sampling"]["random_state"]

        if method == "random_under":
            return RandomUnderSampling(random_state=random_state)
        elif method == "nearmiss":
            return NearMissSampling(version=3)
        elif method == "smote":
            return SMOTESampling()
        elif method == "none":
            return NoSampling()
        else:
            raise ValueError(f"Unknown sampling method: {method}")