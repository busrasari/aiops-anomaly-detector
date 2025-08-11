from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def scatter(df: pd.DataFrame, x: str, y: str, out_path: str | Path):
    plt.figure()
    normal = df[df["anomaly"] == 1]
    anom = df[df["anomaly"] == -1]
    plt.scatter(normal[x], normal[y], s=12, label="normal")
    plt.scatter(anom[x], anom[y], s=22, marker="x", label="anomaly")
    plt.xlabel(x); plt.ylabel(y); plt.legend(); plt.title(f"{x} vs {y}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
