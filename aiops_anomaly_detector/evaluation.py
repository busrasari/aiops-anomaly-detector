import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

def eval_if_labeled(df: pd.DataFrame, y_col="label", y_pred_col="anomaly") -> str | None:
    if y_col in df.columns and y_pred_col in df.columns:
        y_true = df[y_col].to_numpy()
        y_pred = df[y_pred_col].to_numpy()
        return classification_report(y_true, y_pred, digits=3)
    return None

def synthetic(n_normal=200, n_anomalies=10, random_state=42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    normal = rng.normal(0, 1, size=(n_normal, 3))
    anomalies = rng.normal(5, 1, size=(n_anomalies, 3))
    data = np.vstack([normal, anomalies])
    labels = np.hstack([np.ones(n_normal), -1 * np.ones(n_anomalies)])
    df = pd.DataFrame(data, columns=["cpu_usage", "memory_usage", "latency"])
    df["label"] = labels
    return df
