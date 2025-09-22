import pandas as pd
import time
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import os

def cleaner(df):
    if "ignore" in df.columns:
        df = df.drop(columns=["ignore"])
    df["log_return"] = np.log(df["close"]/df["open"])
    df["volume"] = df["volume"].replace(0, np.nan)  # Remplace 0 par NaN
    df["log_vol"] = np.log(df["volume"])
    return df

def detect_high_of_day(df):
    df["date"] = pd.to_datetime(df["closeTime"], utc=True, errors="coerce")
    df["day"] = df["date"].dt.date
    daily_high = df.groupby("day")["close"].transform("max")
    daily_min = df.groupby("day")["close"].transform("min")
    df["is_high_of_day"] = df["close"] == daily_high
    df["is_min_of_day"] = df["close"] == daily_min
    return df

def current_max_low(df):
    df["date"] = pd.to_datetime(df["closeTime"], utc=True, errors="coerce")
    df["day"] = df["date"].dt.date
    df["cummax_close"] = df.groupby("day")["close"].cummax()
    df["cummin_close"] = df.groupby("day")["close"].cummin()
    df["is_current_max"] = df["close"] == df["cummax_close"]
    df["is_current_min"] = df["close"] == df["cummin_close"]
    return df

def mva(df,window):
    df[f'mva_{window}'] = df['close'].rolling(window=window).mean()
    return df

def preparation_data(df, name):
    data = cleaner(df)
    data = mva(data, 20)
    data = mva(data, 50)
    data = mva(data, 100)
    data = current_max_low(data)
    data = detect_high_of_day(data)
    data['target'] = ((data['is_high_of_day']) | (data['is_min_of_day'])).astype(int)
    # Réordonne les colonnes si elles existent
    columns_order = [
        'openTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime',
        'quoteAssetVolume', 'numberOfTrades', 'takerBuyBase', 'takerBuyQuote',
        'date', 'day', 'is_current_max', 'cummin_close', 'is_high_of_day', 'is_min_of_day',
        'log_return', 'log_vol', 'cummax_close', 'mva_20', 'mva_50', 'mva_100', 'is_current_min'
    ]
    data = data[[col for col in columns_order if col in data.columns]]
    # Crée le dossier si besoin
    os.makedirs("../data/cleaned_data", exist_ok=True)
    data.to_csv(f"{name}.csv", index=False)
    return data

from sklearn.preprocessing import StandardScaler

def prepare_for_sgd(df):
    df = cleaner(df)
    df = mva(df, 20)
    df = mva(df, 50)
    df = mva(df, 100)
    df = current_max_low(df)
    df = detect_high_of_day(df)
    # Cible binaire
    df['target'] = ((df['is_high_of_day']) | (df['is_min_of_day'])).astype(int)
    # Supprime les colonnes inutiles
    drop_cols = ['openTime', 'closeTime', 'date', 'day', 'is_high_of_day', 'is_min_of_day']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    # Impute les valeurs manquantes
    df = df.fillna(method='ffill').fillna(method='bfill')
    # Sépare features/target
    X = df.drop(columns=['target'])
    y = df['target']
    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

if __name__ == "__main__":
    df_btc = pd.read_csv("./data/BTCUSDT_15m.csv")
    preparation_data(df_btc, "BTCUSDT_15m")
    df_eth = pd.read_csv("./data/ETHUSDT_15m.csv")
    preparation_data(df_eth, "ETHUSDT_15m")
    df_bnb = pd.read_csv("./data/BNBUSDT_15m.csv")
    preparation_data(df_bnb, "BNBUSDT_15m")
    df_avax = pd.read_csv("./data/AVAXUSDT_15m.csv")
    preparation_data(df_avax, "AVAXUSDT_15m")

    