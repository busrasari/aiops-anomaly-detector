import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def scatter_plot(df, x, y, out_path):
    plt.figure()
    normal = df[df["anomaly"] == 1]
    anom   = df[df["anomaly"] == -1]
    plt.scatter(normal[x], normal[y], s=12, label="normal")
    plt.scatter(anom[x], anom[y], s=22, marker="x", label="anomaly")
    plt.xlabel(x); plt.ylabel(y); plt.legend()
    plt.title(f"Anomaly Scatter: {x} vs {y}")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved scatter plot to {out_path}")

def timeseries_plot(df, time_col, metric, out_path):
    plt.figure()
    t = df[time_col] if time_col in df.columns else df.index
    plt.plot(t, df[metric], linewidth=1)
    anom = df[df["anomaly"] == -1]
    ta = anom[time_col] if time_col in anom.columns else anom.index
    plt.scatter(ta, anom[metric], s=24, marker="x", label="anomaly")
    plt.xlabel(time_col if time_col in df.columns else "index")
    plt.ylabel(metric); plt.legend()
    plt.title(f"Anomaly Time Series: {metric}")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved time series plot to {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with columns incl. 'anomaly'")
    ap.add_argument("--mode", choices=["scatter", "timeseries"], default="scatter")
    ap.add_argument("--x", help="x feature for scatter")
    ap.add_argument("--y", help="y feature for scatter")
    ap.add_argument("--time-col", default=None, help="time column for time series (optional)")
    ap.add_argument("--metric", help="metric to plot for time series")
    ap.add_argument("--output", default="outputs/plot.png")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    Path(Path(args.output).parent).mkdir(parents=True, exist_ok=True)

    if args.mode == "scatter":
        if not args.x or not args.y:
            num = df.select_dtypes(include=["float", "int"]).columns.tolist()
            if len(num) < 2:
                raise ValueError("Need at least two numeric columns for scatter.")
            args.x, args.y = num[0], num[1]
        scatter_plot(df, args.x, args.y, args.output)
    else:
        if not args.metric:
            num = df.select_dtypes(include=["float", "int"]).columns.tolist()
            if not num:
                raise ValueError("Need a numeric column for time series.")
            args.metric = num[0]
        timeseries_plot(df, args.time_col, args.metric, args.output)

if __name__ == "__main__":
    main()
