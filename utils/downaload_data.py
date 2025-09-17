import pandas as pd
import time
import datetime as dt
from binance.spot import Spot as SpotClient
from dateutil.relativedelta import relativedelta

client = SpotClient()  # pas besoin de clés pour les endpoints publics

def fetch_klines_spot_connector(symbol="BTCUSDT", interval="15m", start_ms=None, end_ms=None, limit=1000):                            
    """
    Fetches historical candlestick (kline) data from Binance Spot API and returns it as a pandas DataFrame.

    Args:
        symbol (str): Trading pair symbol (e.g., "BTCUSDT").
        interval (str): Kline interval (e.g., "15m", "1h").
        start_ms (int, optional): Start time in milliseconds since epoch.
        end_ms (int, optional): End time in milliseconds since epoch.
        limit (int): Maximum number of klines to fetch (default: 1000).

    Returns:
        pd.DataFrame: DataFrame containing kline data with columns:
            ['openTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime',
             'quoteAssetVolume', 'numberOfTrades', 'takerBuyBase', 'takerBuyQuote', 'ignore']
            Numeric columns are converted to float, and time columns to datetime (UTC).
    """
    
    rows = client.klines(symbol, interval, startTime=start_ms, endTime=end_ms, limit=limit)
    cols = ["openTime","open","high","low","close","volume","closeTime",
            "quoteAssetVolume","numberOfTrades","takerBuyBase","takerBuyQuote","ignore"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open","high","low","close","volume","quoteAssetVolume","takerBuyBase","takerBuyQuote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["openTime"]  = pd.to_datetime(df["openTime"], unit="ms", utc=True)
    df["closeTime"] = pd.to_datetime(df["closeTime"], unit="ms", utc=True)
    return df

def fetch_klines_full(symbol="BTCUSDT", interval="1h", start="2021-01-01 00:00:00", end="2021-02-01 00:00:00"):
    """
    Récupère tout l'historique entre start et end (UTC) en bouclant par tranches de 1000.
    """
    start_dt = dt.datetime.strptime(start, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
    end_dt   = dt.datetime.strptime(end, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)

    all_dfs = []
    cur_start = start_ms

    while cur_start < end_ms:
        df = fetch_klines_spot_connector(symbol, interval, start_ms=cur_start, end_ms=end_ms, limit=1000)
        if df.empty:
            break
        all_dfs.append(df)
        # avancer d’1 ms après le dernier close récupéré
        last_close = int(df["closeTime"].iloc[-1].timestamp() * 1000)
        cur_start = last_close + 1
        time.sleep(0.2)  # pour éviter le rate limit

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

def save_data(symbol,interval,start,end):
    df = fetch_klines_full(symbol,interval,start,end)
    df.to_csv(f"../data/{symbol}_{interval}.csv",index=False)
    
if __name__ == "__main__":
    symbols = input("Enter the pairs separated by comma (ex: BTCUSDT,ETHUSDT,BNBUSDT): ").strip().upper().split(",")
    interval = input("Enter time interval (ex: 1h, 15m)").strip()
    length = int(input("Enter time horizon from now").strip())
    today = dt.datetime.now()
    start = today - relativedelta(years=length)
    start = start.strftime("%Y-%m-%d %H:%M:%S")
    today = today.strftime("%Y-%m-%d %H:%M:%S")
    for symbol in symbols:
        symbol = symbol.strip()
        print(f"Téléchargement des données pour {symbol} de {start} à {today} en {interval}...")
        save_data(symbol, interval, start, today)

    print("Données sauvegardées dans le dossier ../data/")
