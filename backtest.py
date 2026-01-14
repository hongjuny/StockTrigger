import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import strategy

class Backtester:
    def __init__(self, symbol: str, initial_capital: float = 10000.0, strategy_name: str = "conservative"):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0 # Number of shares
        self.portfolio_history = []
        self.trade_log = []
        self.strategy_name = strategy_name
        
    def fetch_data(self, period="2y", interval="1d"):
        print(f"Fetching {period} data for {self.symbol}...")
        data = yf.download(self.symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        
        # Flatten MultiIndex if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()
        
        # Calculate indicators and signals
        self.df = strategy.add_indicators(data)
        self.df = strategy.calculate_signals_variant(self.df, self.strategy_name)
        return self.df
        
    def run(self):
        if not hasattr(self, 'df'):
            raise ValueError("Data not fetched. Call fetch_data() first.")
            
        print(f"Running backtest (Strategy: {self.strategy_name})...")
        
        in_position = False
        entry_price = 0.0
        
        # Iterate up to len-1 because we trade on "Next Open" (i+1)
        for i in range(0, len(self.df) - 1):
            row = self.df.iloc[i]
            next_row = self.df.iloc[i+1]
            current_date = self.df.index[i]
            exec_date = self.df.index[i+1]
            
            # 1. Check Exit Signal first (if already in position)
            if in_position:
                if row["Sell_Signal"]:
                    # Execute SELL at Next Open
                    price = next_row["Open"]
                    value = self.position * price
                    self.capital = value
                    self.position = 0
                    in_position = False
                    
                    profit = value - (shares * entry_price)
                    self.trade_log.append({
                        "Date": exec_date,
                        "Type": "SELL",
                        "Price": price,
                        "Shares": shares,
                        "Value": value,
                        "Profit": profit,
                        "Reason": "Signal"
                    })

            # 2. Check Entry Signal (if not in position)
            # Use 'elif' if you don't want to buy/sell same day. 
            # Ideally, if we just sold at Open, we *could* buy at Close or next Open. 
            # For simplicity, stay out for the rest of the candle if we just sold.
            if not in_position:
                if row["Buy_Signal"]:
                    # Execute BUY at Next Open
                    price = next_row["Open"]
                    shares = self.capital / price
                    self.position = shares
                    self.capital = 0
                    entry_price = price
                    in_position = True
                    self.trade_log.append({
                        "Date": exec_date,
                        "Type": "BUY",
                        "Price": price,
                        "Shares": shares,
                        "Value": shares * price,
                        "Reason": "Signal"
                    })
            
            # Track Portfolio Value (Mark to Market using Close of i+1)
            # Note: We used i+1 Open for execution, so for daily valuation we use i+1 Close
            current_val = self.capital + (self.position * next_row["Close"])
            self.portfolio_history.append({
                "Date": exec_date,
                "Value": current_val
            })
            
        self.results = pd.DataFrame(self.portfolio_history).set_index("Date")
        print("Backtest complete.")
        
    def plot_results(self):
        if not hasattr(self, 'results'):
            return
            
        plt.figure(figsize=(12, 6))
        plt.plot(self.results.index, self.results["Value"], label="Portfolio Strategy")
        
        # Compare with Buy & Hold
        initial_price = self.df["Close"].iloc[50]
        final_price = self.df["Close"].iloc[-1]
        bnh_return = (self.df["Close"] / initial_price) * self.initial_capital
        # Align B&H with results index
        bnh_return = bnh_return.loc[self.results.index]
        
        plt.plot(bnh_return.index, bnh_return, label="Buy & Hold", alpha=0.5, linestyle="--")
        
        plt.title(f"Backtest Results: {self.symbol}")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def stats(self):
        if not self.trade_log:
            print("No trades made.")
            return
            
        trades = pd.DataFrame(self.trade_log)
        sells = trades[trades["Type"] == "SELL"]
        
        if sells.empty:
            print("No completed trades.")
            return
            
        total_trades = len(sells)
        wins = len(sells[sells["Profit"] > 0])
        win_rate = (wins / total_trades) * 100
        
        total_return = (self.results["Value"].iloc[-1] - self.initial_capital) / self.initial_capital * 100
        
        # MDD
        roll_max = self.results["Value"].cummax()
        drawdown = (self.results["Value"] - roll_max) / roll_max
        max_drawdown = drawdown.min() * 100
        
        print("="*30)
        print(f"Strategy Performance ({self.symbol})")
        print("="*30)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Value:     ${self.results['Value'].iloc[-1]:,.2f}")
        print(f"Total Return:    {total_return:.2f}%")
        print(f"Max Drawdown:    {max_drawdown:.2f}%")
        print(f"Total Trades:    {total_trades}")
        print(f"Win Rate:        {win_rate:.1f}%")
        print("="*30)

if __name__ == "__main__":
    bt = Backtester("TQQQ", 10000, strategy_name="conservative")
    bt.fetch_data(period="2y")
    bt.run()
    bt.stats()
    # bt.plot_results() # Uncomment to show plot
