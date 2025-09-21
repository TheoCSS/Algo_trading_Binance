import pandas as pd
import time
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

def cleaner(df):
    df = df.drop(columns=["ignore"])
    df["log_return"] = np.log(df["close"]/df["open"])
    df["volume"] = df["volume"].replace(0, np.nan)  # Remplace 0 par NaN
    df["log_vol"] = np.log(df["volume"])
    df["openTime"] = df["openTime"]
    return df

def detect_high_of_day(df):
    # S'assurer que la colonne date est bien en datetime
    df["date"] = pd.to_datetime(df["closeTime"], utc=True, errors="coerce")
    # Extraire la date (sans l'heure)
    df["day"] = df["date"].dt.date
    # Trouver le prix max de chaque journée
    daily_high = df.groupby("day")["close"].transform("max")
    daily_min = df.groupby("day")["close"].transform("min")
    # Créer une colonne booléenne : True si c'est le plus haut de la journée
    df["is_high_of_day"] = df["close"] == daily_high
    df["is_min_of_day"] = df["close"] == daily_min
    return df

def current_max_low(df):
    # S'assurer que la colonne date est bien en datetime
    df["date"] = pd.to_datetime(df["closeTime"], utc=True, errors="coerce")
    df["day"] = df["date"].dt.date
    # Rolling max et min du close pour chaque jour
    df["cummax_close"] = df.groupby("day")["close"].cummax()
    df["cummin_close"] = df.groupby("day")["close"].cummin()
    # True si le close actuel est le max ou min du jour jusqu'à présent
    df["is_current_max"] = df["close"] == df["cummax_close"]
    df["is_current_min"] = df["close"] == df["cummin_close"]
    return df

def preparation_data(df,name):
    data = cleaner(df)
    data = mva(data,20)
    data = mva(data,50)
    data = mva(data,100)
    data = current_max_low(data)
    data = detect_high_of_day(data)
    data = data.columns(['openTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime',
       'quoteAssetVolume', 'numberOfTrades', 'takerBuyBase', 'takerBuyQuote', 
       'date', 'day', 'is_current_max', 'cummin_close', 'is_high_of_day', 'is_min_of_day'])
    data.to_csv(f"../data/cleaned_data/{name}.csv", index=False)
    return data