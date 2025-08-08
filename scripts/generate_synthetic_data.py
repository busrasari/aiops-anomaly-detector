import pandas as pd
import numpy as np
from pathlib import Path

def generate_data(n_normal=200, n_anomalies=10, random_state=42):
    np.random.seed(random_state)

    # Normal veriler (mean=0, std=1)
    normal_data = np.random.normal(0, 1, size=(n_normal, 3))

    # Anomaliler (mean=5, std=1)
    anomalies = np.random.normal(5, 1, size=(n_anomalies, 3))

    data = np.vstack([normal_data, anomalies])
    labels = np.hstack([np.ones(n_normal), -1 * np.ones(n_anomalies)])  # 1=normal, -1=anomali

    df = pd.DataFrame(data, columns=["cpu_usage", "memory_usage", "latency"])
    df["label"] = labels
    return df

if __name__ == "__main__":
    output_path = Path("data/synthetic.csv")
    df = generate_data()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Synthetic dataset saved to {output_path}")
