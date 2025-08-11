from pathlib import Path
from aiops_anomaly_detector.evaluation import synthetic
from aiops_anomaly_detector.io import save_csv

if __name__ == "__main__":
    df = synthetic()
    save_csv(df, Path("data/synthetic.csv"))
    print("✅ Synthetic dataset saved to data/synthetic.csv")