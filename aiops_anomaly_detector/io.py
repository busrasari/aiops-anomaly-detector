from pathlib import Path
import pandas as pd

def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)

def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)