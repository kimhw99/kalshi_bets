from private_key import load_private_key, get, post
import requests
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from load_data import load_data

from environment import StockTradingEnv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
import torch as th

import gradio as gr
import pandas as pd

def get_bets():
    eastern_tz = ZoneInfo("America/New_York")
    hour_from_now = datetime.now(eastern_tz) + timedelta(hours=1)
    event_ticker = f"kxbtcd-{hour_from_now.strftime('%y')}{datetime.now().strftime('%b').lower()}{hour_from_now.day:0{2}d}{hour_from_now.hour:0{2}d}".upper()
    
    url = f"https://api.elections.kalshi.com/trade-api/v2/markets?event_ticker={event_ticker}&status=open"
    response = requests.get(url)
    

    """
    tickers = []

    for market in response.json()['markets']:
        ticker = market['ticker']
        tickers.append(ticker)

    return tickers
    """

    return response.json()['markets']

def make_bets(env, model, model_vector):

    obs, _ = env.reset()

    with th.no_grad():
        action, _ = model.predict(obs, deterministic=True)
        change_perc, _ = model_vector.predict(obs)

        agreement = (action.item() * change_perc.item() > 0)
        change_perc = (change_perc.item()**3)/20 + 1
    
    action = action.item()
    cur_price = env.stock_price_history[env.current_step]
    pred_price = env.stock_price_history[env.current_step-1] * change_perc
    
    # Get Possible Bets
    bets = get_bets()

    print(f'Current Price: {cur_price}')
    print(f'Predicted Price: {pred_price}')
    print()
    
    result = pd.DataFrame(columns=['Event', 'Yes Bids', 'No Bids', 'Bet', 'Floor', 'Bet Amount'])

    for bet in bets:
    
        bet_price = bet['floor_strike']
        bet_amount = 0
    
        # Action
        side = None
        if action >= 0: # Pred value goes up
            if bet_price < cur_price:
                side = "Yes"
                bet_amount = (bet['yes_bid'] + 0.01) / (bet['yes_bid'] + bet['no_bid'] + 0.01)

            else:
                if agreement:
                    if bet_price < pred_price:
                        side = "Yes"
                        bet_amount = (bet['yes_bid'] + 0.01) / (bet['yes_bid'] + bet['no_bid'] + 0.01)
        
                    else:
                        side= "No"
                        bet_amount = (bet['no_bid'] + 0.01) / (bet['yes_bid'] + bet['no_bid'] + 0.01)

        else: # Pred value goes down
            if bet_price > cur_price:
                side = "No"
                bet_amount = (bet['no_bid'] + 0.01) / (bet['yes_bid'] + bet['no_bid'] + 0.01)

            else:
                if agreement:
                    if bet_price > pred_price:
                        side = "No"
                        bet_amount = (bet['no_bid'] + 0.01) / (bet['yes_bid'] + bet['no_bid'] + 0.01)
        
                    else:
                        side = "Yes"
                        bet_amount = (bet['yes_bid'] + 0.01) / (bet['yes_bid'] + bet['no_bid'] + 0.01)
        
        # Take Bet
        #print(f'{bet['ticker']}')
        #print(f"{bet['floor_strike']}$ or above")
        #print(f'Yes {bet['yes_bid']} / No { bet['no_bid']} / Bet {side}')
        #print()
        

        new_df = pd.DataFrame({'Event': bet['ticker'], 'Yes Bids': bet['yes_bid'], 'No Bids': bet['no_bid'], 'Bet': side, 'Floor': bet['floor_strike'], 'Bet Amount': bet_amount}, index=[0])
        result = pd.concat([result, new_df], ignore_index=True)

    return result

if __name__ == "__main__":
    print("Hello World")

    # kalshi api
    with open('kalshi_api_key.txt', 'r') as file:
        API_KEY_ID = file.read()
    PRIVATE_KEY_PATH = 'kalshi-pro-key.txt'
    private_key = load_private_key(PRIVATE_KEY_PATH)
    demo = False

    # coingecko api
    crypto = 'bitcoin'

    with open('coingecko_api_key.txt', 'r') as file:
        api_key = file.read()

    # Models
    model = PPO.load("pred_dir.zip")
    model_vector = PPO.load("pred_vector.zip")

    # Data & Observation Load
    df = load_data(crypto, api_key)
    env = StockTradingEnv(df[-168:])
    obs, _ = env.reset()
    
    print(make_bets(env, model, model_vector))
