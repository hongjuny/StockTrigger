import pandas as pd
import yfinance as yf
import strategy
from backtest import Backtester

def grid_search_supertrend(symbol="TQQQ", period="10y"):
    print(f"Fetching data for {symbol}...")
    # Fetch data ONCE
    raw_data = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=False)
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)
    raw_data = raw_data[["Open", "High", "Low", "Close", "Volume"]].dropna()

    # Define Parameter Grid (Fine-Tuning)
    # Previous Winner: Period 10, Multiplier 2.5
    # Zooming in around these values
    periods = [8, 9, 10, 11, 12]
    multipliers = [2.3, 2.4, 2.5, 2.6, 2.7]
    
    results = []
    
    # We need to modify strategy.add_indicators dynamically or calculate ST here.
    # Since strategy.add_indicators is hardcoded, we will calculate ST locally in this loop for speed.
    
    print(f"Testing {len(periods) * len(multipliers)} combinations (Fine-Tuning)...")
    
    for p in periods:
        for m in multipliers:
            # 1. Calculate SuperTrend with current params
            df = raw_data.copy()
            
            # ATR Calculation
            high = df["High"]
            low = df["Low"]
            close = df["Close"]
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.ewm(alpha=1/p, adjust=False).mean()
            
            # Bands
            basic_upper = (high + low) / 2 + m * atr
            basic_lower = (high + low) / 2 - m * atr
            
            final_upper = [0.0] * len(df)
            final_lower = [0.0] * len(df)
            trend = [1] * len(df)
            
            close_vals = close.values
            bu_vals = basic_upper.values
            bl_vals = basic_lower.values
            
            # Numba would be faster, but this is short enough for daily data
            for i in range(1, len(df)):
                # Final Upper
                if bu_vals[i] < final_upper[i-1] or close_vals[i-1] > final_upper[i-1]:
                    final_upper[i] = bu_vals[i]
                else:
                    final_upper[i] = final_upper[i-1]
                
                # Final Lower
                if bl_vals[i] > final_lower[i-1] or close_vals[i-1] < final_lower[i-1]:
                    final_lower[i] = bl_vals[i]
                else:
                    final_lower[i] = final_lower[i-1]
                
                # Trend
                if trend[i-1] == 1:
                    if close_vals[i] < final_lower[i]:
                        trend[i] = -1
                    else:
                        trend[i] = 1
                else:
                    if close_vals[i] > final_upper[i]:
                        trend[i] = 1
                    else:
                        trend[i] = -1
            
            # 2. Backtest Logic (Vectorized for speed)
            # Signal: Buy when trend flips to 1, Sell when trend flips to -1
            trend_series = pd.Series(trend, index=df.index)
            # Shift trend by 1 to represent "Yesterday's Trend" used for Today's decision? 
            # No, standard ST uses current close to flip. We trade Next Open.
            
            # Buy Signal: Yesterday ended with Trend 1, Day Before was -1.
            # Trade at Today Open.
            
            # trends[i] is the trend at Close of day i.
            # if trends[i] == 1 and trends[i-1] == -1: Buy at Open of i+1
            
            signals = pd.Series(0, index=df.index) # 1=Buy, -1=Sell
            
            # Identify flip days
            # flip_bull: Trend becomes 1 today
            flip_bull = (trend_series == 1) & (trend_series.shift(1) == -1)
            flip_bear = (trend_series == -1) & (trend_series.shift(1) == 1)
            
            # Execution prices (Next Open)
            # We filter for valid dates (up to len-1)
            
            # Quick Vectorized Backtest
            # We need to iterate to handle "In Position" state properly
            
            capital = 10000.0
            shares = 0
            in_pos = False
            trade_count = 0
            
            # Convert to lists for speed
            opens = df["Open"].values
            flips_b = flip_bull.values
            flips_s = flip_bear.values
            
            # Start loop
            for i in range(len(df) - 1):
                # Check for signals generated at Close of i, executed at Open of i+1
                if not in_pos and flips_b[i]:
                    # Buy
                    price = opens[i+1]
                    shares = capital / price
                    capital = 0
                    in_pos = True
                    trade_count += 1
                elif in_pos and flips_s[i]:
                    # Sell
                    price = opens[i+1]
                    capital = shares * price
                    shares = 0
                    in_pos = False
                    trade_count += 1
            
            # Final Value
            final_val = capital + (shares * df["Close"].iloc[-1])
            ret = (final_val - 10000) / 10000 * 100
            
            # MDD approximation (Daily Close equity curve)
            # Construct equity curve
            # This is complex to do fully vectorized with trades, but we can approximate:
            # If in market, return follows TQQQ, else 0.
            # Let's skip MDD for now or do a rough calculation if critical.
            # User cares about MDD. Let's try to capture it.
            
            # Reconstruct equity curve
            equity = []
            curr_cap = 10000.0
            curr_shares = 0
            curr_pos = False
            
            for i in range(len(df)):
                if i > 0:
                    # Execute pending orders from i-1
                    if not curr_pos and flips_b[i-1]:
                        price = opens[i]
                        curr_shares = curr_cap / price
                        curr_cap = 0
                        curr_pos = True
                    elif curr_pos and flips_s[i-1]:
                        price = opens[i]
                        curr_cap = curr_shares * price
                        curr_shares = 0
                        curr_pos = False
                
                # Mark to market (Close)
                val = curr_cap + (curr_shares * close_vals[i])
                equity.append(val)
            
            equity_s = pd.Series(equity)
            dd = (equity_s - equity_s.cummax()) / equity_s.cummax()
            mdd = dd.min() * 100
            
            results.append({
                "Period": p,
                "Multiplier": m,
                "Return": ret,
                "MDD": mdd,
                "Trades": trade_count // 2 # Round trips
            })
            
    # Sort and Display
    res_df = pd.DataFrame(results).sort_values(by="Return", ascending=False)
    print("\n" + "="*60)
    print(f"SuperTrend Fine-Tuning Results (TQQQ, {period})")
    print("="*60)
    print(res_df.to_string(index=False, float_format="%.2f"))
    print("="*60)
    
    # Save to CSV
    res_df.to_csv("grid_search_results.csv", index=False)

if __name__ == "__main__":
    grid_search_supertrend()