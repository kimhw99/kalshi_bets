import pandas as pd
import requests
from datetime import datetime

def update_money():
    with open('money.txt', 'r') as f:
        money = float(f.read())

    df = pd.read_csv("bets.csv")

    url = "https://api.coingecko.com/api/v3/simple/price?vs_currencies=usd&ids=bitcoin&names=Bitcoin&symbols=btc"

    with open('coingecko_api_key.txt', 'r') as file:
        api_key = file.read()

    headers = {"x-cg-demo-api-key": api_key}
    response = requests.get(url, headers=headers)

    price = response.json()["bitcoin"]["usd"]

    for i in range(len(df)):

        if df['Bet'][i] == "Yes":

            payout = (df['Yes Bids'][i] + df['No Bids'][i] + df['Bet Amount'][i]) / (df['Yes Bids'][i] + df['Bet Amount'][i]) 
            # print(payout)

            if df['Floor'][i] < price:
                money -= df['Bet Amount'][i]
                money += payout
            
            else:
                money -= df['Bet Amount'][i]
        
        elif df['Bet'][i] == "No":

            payout = (df['Yes Bids'][i] + df['No Bids'][i] + df['Bet Amount'][i]) / (df['No Bids'][i] + df['Bet Amount'][i])
            # print(payout)

            if df['Floor'][i] > price:
                money -= df['Bet Amount'][i]
                money += payout

            else:
                money -= df['Bet Amount'][i]

    with open('money.txt', "w") as file:
        file.write(str(money))

    with open(f'logs/money/{datetime.now()}.txt', "w") as file:
        file.write(str(money))

    df.to_csv(f'logs/bets/{datetime.now()}.csv')

    return money

