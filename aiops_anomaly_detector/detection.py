import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def select_numeric(df: pd.DataFrame) -> pd.DataFrame:
    X = df.select_dtypes(include=["float", "int"])
    if X.empty:
        raise ValueError("No numeric columns found to run anomaly detection.")
    return X

def detect_isolation_forest(X: pd.DataFrame, contamination: float, random_state: int = 42):
    model = IsolationForest(contamination=contamination, random_state=random_state)
    preds = model.fit_predict(X)
    scores = model.decision_function(X)
    return preds, scores

def detect_lof(X: pd.DataFrame, contamination: float, n_neighbors: int = 20):
    model = LocalOutlierFactor(contamination=contamination, n_neighbors=n_neighbors)
    preds = model.fit_predict(X)
    scores = model.negative_outlier_factor_
    return preds, scores
