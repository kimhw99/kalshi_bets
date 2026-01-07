import requests
from datetime import datetime, timedelta
import pandas as pd

# Load Bitcoin data from the past 7 days to feed into the PPO model
# Given in Hourly Intervals
# https://docs.coingecko.com/v3.0.1/reference/coins-id-market-chart-range


def load_data(crypto, api_key, days=7):
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    end_time = int(end_time.timestamp())
    start_time = int(start_time.timestamp())

    call = f"https://api.coingecko.com/api/v3/coins/{crypto}/market_chart/range?vs_currency=usd&from={start_time}&to={end_time}&x_cg_demo_api_key={api_key}"
    response = requests.get(call).json()

    df = pd.DataFrame(columns=['Timestamp', 'Value'])

    for i in range(len(response['prices'])):
        timestamp = datetime.fromtimestamp(response['prices'][i][0]/1000)
        value = response['prices'][i][1]
        df2 = pd.DataFrame([[timestamp, value]], columns=['Timestamp', 'Value'])
        df = pd.concat([df, df2], ignore_index=True)
    
    return df

