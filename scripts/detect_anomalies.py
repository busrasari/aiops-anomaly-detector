import pandas as pd
from sklearn.ensemble import IsolationForest

# Örnek veri yolu (veri dosyasını data/ altına koymalısın)
DATA_PATH = "data/sample.csv"

def load_data(path):
    return pd.read_csv(path)

def detect_anomalies(df):
    model = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = model.fit_predict(df.select_dtypes(include=['float', 'int']))
    return df

def main():
    try:
        df = load_data(DATA_PATH)
        print("📊 Veri başarıyla yüklendi.")
        result = detect_anomalies(df)
        print(result.head())
        print("\n🚨 Anomali sayısı:", (result["anomaly"] == -1).sum())
    except FileNotFoundError:
        print(f"❌ Veri dosyası bulunamadı: {DATA_PATH}")
    except Exception as e:
        print(f"⚠️ Hata: {e}")

if __name__ == "__main__":
    main()
