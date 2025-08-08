import argparse
import logging
from pathlib import Path
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report

def setup_logger(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "run.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())

def load_data(path: Path) -> pd.DataFrame:
    logging.info(f"Loading data from {path}")
    return pd.read_csv(path)

def detect_isolation_forest(X: pd.DataFrame, contamination: float, random_state: int = 42):
    model = IsolationForest(contamination=contamination, random_state=random_state)
    preds = model.fit_predict(X)
    scores = model.decision_function(X)
    return preds, scores

def detect_lof(X: pd.DataFrame, contamination: float, n_neighbors: int = 20):
    model = LocalOutlierFactor(contamination=contamination, n_neighbors=n_neighbors)
    preds = model.fit_predict(X)                 # -1 anomalı, 1 normal
    scores = model.negative_outlier_factor_      # daha küçük = daha anormal
    return preds, scores

def main():
    parser = argparse.ArgumentParser(description="AIOps anomaly detection")
    parser.add_argument("--input", required=True, help="CSV input path (e.g., data/sample.csv)")
    parser.add_argument("--model", default="isoforest", choices=["isoforest", "lof"])
    parser.add_argument("--contamination", type=float, default=0.05, help="Expected anomaly ratio")
    parser.add_argument("--output-dir", default="outputs", help="Directory to write results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(Path("logs"))

    try:
        df = load_data(Path(args.input))
        num_df = df.select_dtypes(include=["float", "int"])
        if num_df.empty:
            raise ValueError("No numeric columns found to run anomaly detection.")

        if args.model == "isoforest":
            preds, scores = detect_isolation_forest(num_df, args.contamination)
        else:
            preds, scores = detect_lof(num_df, args.contamination)

        result = df.copy()
        result["anomaly"] = preds            # 1 normal, -1 anomali
        result["score"] = scores

        out_csv = output_dir / "anomalies.csv"
        result.to_csv(out_csv, index=False)
        logging.info(f"Saved results to {out_csv}")

        if "label" in result.columns:
        y_true = result["label"]
        y_pred = result["anomaly"]
        print("\n📊 Evaluation Metrics:")
        print(classification_report(y_true, y_pred, digits=3))

        anom_count = (result["anomaly"] == -1).sum()
        total = len(result)
        logging.info(f"Anomalies: {anom_count}/{total} (contamination={args.contamination})")

    except Exception as e:
        logging.exception(f"Run failed: {e}")
        raise

if __name__ == "__main__":
    main()