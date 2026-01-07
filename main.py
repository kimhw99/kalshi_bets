import gradio as gr
import pandas as pd
import numpy as np
import time
from load_data import load_data

from environment import StockTradingEnv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
import torch as th
from datetime import datetime, timedelta

from make_bets import make_bets
from private_key import load_private_key

import random
import threading

from update_money import update_money

import schedule
import time

# Function to simulate fetching new data

def load_predicted_data(df, model_vector, hours_lookahead=5, return_timesteps=24):

    for h in range(hours_lookahead):
        env = StockTradingEnv(df[-168:])
        next_timestamp = df['Timestamp'][-1:].item() + timedelta(hours=1)
        obs, _ = env.reset()

        with th.no_grad():
            change_perc, _ = model_vector.predict(obs)
            change_perc = (change_perc.item()**3)/20 + 1
        
        pred_price = env.stock_price_history[env.current_step-1] * change_perc

        df.loc[len(df)] = [next_timestamp, pred_price]

    df_normal = df.iloc[:-hours_lookahead].copy()
    df_normal['Type'] = "Historical"
    
    df_special = df.iloc[-hours_lookahead-1:].copy()
    df_special['Type'] = "Predicted"
    
    df = pd.concat([df_normal, df_special])

    df["Timestamp"] = df["Timestamp"].dt.tz_localize('US/Eastern')#.dt.tz_convert("US/Eastern")

    return df[-return_timesteps:].reset_index()


if __name__ == "__main__":

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
    
    # Bunch of display Functions
    def btc_graph_data():
        return load_predicted_data(load_data(crypto, api_key), model_vector)
    
    def bets_display():
        positions = pd.read_csv("bets.csv")
        return positions
    
    def money_amount():
        with open('money.txt', 'r') as f:
            money = f.read()
        return str(money)
    
    def bets(env=env, model=model, model_vector=model_vector):
        
        def task_0():
            return update_money()

        def task_1():
            positions = make_bets(env, model, model_vector)
            positions.to_csv("bets.csv", index=False)

        schedule.every().hour.at(":00").do(task_0)
        schedule.every().hour.at(":01").do(task_1)

        for i in range(1, 60):
            schedule.every().hour.at(f":{i:02}").do(task_1)

        while True:
            schedule.run_pending()
            time.sleep(1)

    # Initialize Bets
    positions = make_bets(env, model, model_vector)
    positions.to_csv("bets.csv", index=False)

    t1 = threading.Thread(target=bets)
    t1.start()
   
    with gr.Blocks() as demo:
        gr.Markdown("BTC Price History")

        plot = gr.LinePlot(
            btc_graph_data,
            x="Timestamp", 
            y="Value", 
            color="Type",
            every=300,
            height=300
        )

        money_display = gr.Textbox(label="Money", 
                                  value=money_amount(), 
                                  interactive=False)

        timer = gr.Timer(1.0)
        timer.tick(money_amount, outputs=money_display)

        gr.Dataframe(value=bets_display, every=10)

    demo.launch(inbrowser=True)