import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

# Note: Base Environment Generated with Gemini 3
# Both environments are used in jupyter notebooks in the /train directory
# To train PPO models for BTC forecasting


# Dynamic Environment
# State: Past N Percentage Differences between Bitcoin Prices & General Trendline
# Action: (-1 ~ 1) (determine percentage of stocks (BTC) to buy or sell in current step)

class StockTradingEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, df, initial_balance=10000):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        
        self.stock_price_history = df['Value'].values

        # Action Space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observation Space: (N closing prices, shares held, cash balance)
        # N=5, so shape is 5 + 2 = 7
        self.n = 24 * 7
        obs_len = self.n
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)

        self.current_step = max(0, self.n-1)
        
        # State tracking (will be reset in self.reset())
        self.balance = self.initial_balance
        self.shares_held = 0
        self.portfolio_value = self.initial_balance
        self.last_portfolio_value = self.initial_balance
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = max(0, self.n-1) 
        self.last_portfolio_value = self.initial_balance
        
        # Get the initial observation (e.g., first n closing prices, 0 shares, initial balance)
        observation = self._get_observation().astype(np.float32)
        info = self._get_info()
        return observation, info

    def _get_observation(self):
        # A simple observation: last n close prices, shares held, current balance
        
        start = self.current_step - (self.n-1) if self.current_step >= (self.n-1) else 0
        prices = self.stock_price_history[start:self.current_step + 1]
        trend = ((prices[0] - prices[-1]) / prices[0]) * 100
        
        # Pad with zeros if less than n steps
        padded_prices = np.pad(prices, (self.n - len(prices), 0), 'constant', constant_values=prices[0])

        change = []
        for i in range(len(padded_prices)-1):
            change.append(((padded_prices[i] - padded_prices[i+1])/padded_prices[i])*100)
        
        return np.append(change, [trend]).astype(np.float32)

    def _get_info(self):
        # Calculate current total portfolio value
        current_price = self.stock_price_history[self.current_step]
        self.portfolio_value = self.balance + self.shares_held * current_price
        
        return {"portfolio_value": self.portfolio_value}
    
    def step(self, action, amount=0.001):
        self.current_step += 1
        current_price = self.stock_price_history[self.current_step]
        reward = 0
        
        if action >= 0: # Buy
            # Simple approach: Buy a fixed number of shares if cash is available
            shares_to_buy = (self.balance * action) / current_price
            cost = shares_to_buy * current_price
            if self.balance >= cost:
                self.shares_held += shares_to_buy
                self.balance -= cost
                
        elif action < 0: # Sell
            # Simple approach: Sell a fixed number of shares if held
            shares_to_sell = action * self.shares_held * -1
            if self.shares_held >= shares_to_sell:
                self.shares_held -= shares_to_sell
                self.balance += shares_to_sell * current_price

        # Calculate new portfolio value and reward
        new_portfolio_value = self.balance + self.shares_held * current_price
        reward = new_portfolio_value - self.last_portfolio_value
        self.last_portfolio_value = new_portfolio_value

        # Check for termination
        terminated = self.current_step >= len(self.stock_price_history) - 1
        truncated = False # No time limit termination
        
        observation = self._get_observation().astype(np.float32)
        info = self._get_info()
        
        return observation, float(reward.item()), terminated, truncated, info

    def render(self):
        # Simple print for demonstration
        print(f"Step: {self.current_step} | Price: {self.stock_price_history[self.current_step]:.2f} | Balance: {self.balance:.2f} | Shares: {self.shares_held} | Value: {self.portfolio_value:.2f}")

    def close(self):
        # Clean up resources if necessary
        pass



# A seperate environment with different rewards was used to predict prices directly
# Action: [-1, 1] Predict percentage change
"""
class StockTradingEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, df, initial_balance=10000):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        
        self.stock_price_history = df['Value'].values

        # Action Space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observation Space: (N closing prices, shares held, cash balance)
        # N=5, so shape is 5 + 2 = 7
        self.n = 24 * 7
        obs_len = self.n
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)

        self.current_step = max(0, self.n-1)
        
        # State tracking (will be reset in self.reset())
        self.balance = self.initial_balance
        self.shares_held = 0
        self.portfolio_value = self.initial_balance
        self.last_portfolio_value = self.initial_balance
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = max(0, self.n-1) 
        self.last_portfolio_value = self.initial_balance
        
        # Get the initial observation (e.g., first n closing prices, 0 shares, initial balance)
        observation = self._get_observation().astype(np.float32)
        info = self._get_info()
        return observation, info

    def _get_observation(self):
        # A simple observation: last n close prices, shares held, current balance
        
        start = self.current_step - (self.n-1) if self.current_step >= (self.n-1) else 0
        prices = self.stock_price_history[start:self.current_step + 1]
        trend = ((prices[0] - prices[-1]) / prices[0]) * 100 # change percentage within window
        
        # Pad with zeros if less than n steps
        padded_prices = np.pad(prices, (self.n - len(prices), 0), 'constant', constant_values=prices[0])

        change = [] # change percentage between timesteps 
        for i in range(len(padded_prices)-1):
            change.append(((padded_prices[i] - padded_prices[i+1])/padded_prices[i])*100)
        
        return np.append(change, [trend]).astype(np.float32)

    def _get_info(self):
        # Calculate current total portfolio value
        current_price = self.stock_price_history[self.current_step]
        self.portfolio_value = self.balance + self.shares_held * current_price
        
        return {"portfolio_value": self.portfolio_value}
    
    def step(self, action, amount=0.001):
        self.current_step += 1
        
        current_price = self.stock_price_history[self.current_step]
        last_price = self.stock_price_history[self.current_step-1]

        pred_price = last_price * ((action**3)/20 + 1)
        
        # Calculate new portfolio value and reward
        new_portfolio_value = self.balance + self.shares_held * current_price
        reward = (abs(current_price - pred_price) / current_price) * -100
        self.last_portfolio_value = new_portfolio_value

        # Check for termination
        terminated = self.current_step >= len(self.stock_price_history) - 1
        truncated = False # No time limit termination
        
        observation = self._get_observation().astype(np.float32)
        info = self._get_info()
        
        return observation, float(reward.item()), terminated, truncated, info

    def render(self):
        # Simple print for demonstration
        print(f"Step: {self.current_step} | Price: {self.stock_price_history[self.current_step]:.2f} | Balance: {self.balance:.2f} | Shares: {self.shares_held} | Value: {self.portfolio_value:.2f}")

    def close(self):
        # Clean up resources if necessary
        pass
"""