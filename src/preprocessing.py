import sys
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PowerFactorCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer to clean invalid power factor values (< 0 or > 100).
    Replaces values outside [0, 100] with NaN for downstream median imputation.
    """
    def __init__(self, columns=None):
        self.columns = columns if columns is not None else [
            "Lagging_Current_Power_Factor",
            "Leading_Current_Power_Factor"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.columns is not None:
            for col in self.columns:
                if col in X.columns:
                    X.loc[
                        (X[col] < 0) | (X[col] > 100),
                        col
                    ] = np.nan
        return X


# Compatibility shim: if the model was serialized when PowerFactorCleaner was in __main__,
# this ensures joblib unpickling can resolve PowerFactorCleaner under __main__.
if "__main__" in sys.modules and not hasattr(sys.modules["__main__"], "PowerFactorCleaner"):
    setattr(sys.modules["__main__"], "PowerFactorCleaner", PowerFactorCleaner)