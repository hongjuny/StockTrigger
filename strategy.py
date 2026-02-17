import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class TriggerResult:
    message: str
    color: str

def identify_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Identifies extended candlestick patterns without TA-Lib."""
    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    
    # 0. Basic Metrics
    body = c - o
    abs_body = body.abs()
    candle_range = h - l
    midpoint = (o + c) / 2
    
    # Previous candles
    o1, h1, l1, c1 = o.shift(1), h.shift(1), l.shift(1), c.shift(1)
    o2, h2, l2, c2 = o.shift(2), h.shift(2), l.shift(2), c.shift(2)
    
    body1 = c1 - o1
    abs_body1 = body1.abs()
    midpoint1 = (o1 + c1) / 2
    
    abs_body2 = (c2 - o2).abs()
    
    # Helper conditions
    is_green = c > o
    is_red = c < o
    is_green1 = c1 > o1
    is_red1 = c1 < o1
    is_green2 = c2 > o2
    is_red2 = c2 < o2
    
    # Small body definition (e.g., < 30% of average range, simplified here as < 50% of current range)
    is_small = abs_body < (candle_range * 0.4)
    is_small1 = abs_body1 < ((h1 - l1) * 0.4)
    
    # Long body definition
    is_long = abs_body > (candle_range * 0.6)
    is_long1 = abs_body1 > ((h1 - l1) * 0.6)
    is_long2 = abs_body2 > ((h2 - l2) * 0.6)

    # --- 1-Candle Patterns ---
    
    # Hammer / Shooting Star
    lower_shadow = pd.concat([c, o], axis=1).min(axis=1) - l
    upper_shadow = h - pd.concat([c, o], axis=1).max(axis=1)
    
    df["Pattern_Hammer"] = (lower_shadow > 2 * abs_body) & (upper_shadow < abs_body)
    df["Pattern_Shooting_Star"] = (upper_shadow > 2 * abs_body) & (lower_shadow < abs_body)
    df["Pattern_Doji"] = abs_body <= (candle_range * 0.1)

    # --- 2-Candle Patterns ---
    
    # Engulfing
    # Bull: Prev Red, Curr Green, Open < PrevClose, Close > PrevOpen
    df["Pattern_Bull_Engulfing"] = is_red1 & is_green & (o < c1) & (c > o1)
    # Bear: Prev Green, Curr Red, Open > PrevClose, Close < PrevOpen
    df["Pattern_Bear_Engulfing"] = is_green1 & is_red & (o > c1) & (c < o1)
    
    # Harami (Inside Bar)
    # Bull: Prev Long Red, Curr Small Green inside Prev Body
    df["Pattern_Bull_Harami"] = is_long1 & is_red1 & is_green & (o > c1) & (c < o1)
    # Bear: Prev Long Green, Curr Small Red inside Prev Body
    df["Pattern_Bear_Harami"] = is_long1 & is_green1 & is_red & (o < c1) & (c > o1)
    
    # Piercing Line (Bullish Reversal)
    # Prev Long Red, Curr Long Green, Open < Prev Low (or Close), Close > Prev Midpoint
    df["Pattern_Piercing"] = is_long1 & is_red1 & is_long & is_green & (o < c1) & (c > midpoint1)
    
    # Dark Cloud Cover (Bearish Reversal)
    # Prev Long Green, Curr Long Red, Open > Prev High (or Close), Close < Prev Midpoint
    df["Pattern_Dark_Cloud"] = is_long1 & is_green1 & is_long & is_red & (o > c1) & (c < midpoint1)

    # --- 3-Candle Patterns ---
    
    # Morning Star (Bullish Reversal)
    # Long Red -> Small Gap Down -> Long Green > Midpoint of 1st
    cond_ms = is_long2 & is_red2 & is_small1 & is_long & is_green
    cond_ms_gap = (pd.concat([c1, o1], axis=1).max(axis=1) < c2) # Body of 2 below Close of 1
    cond_ms_close = c > ((o2 + c2) / 2)
    df["Pattern_Morning_Star"] = cond_ms & cond_ms_gap & cond_ms_close
    
    # Evening Star (Bearish Reversal)
    # Long Green -> Small Gap Up -> Long Red < Midpoint of 1st
    cond_es = is_long2 & is_green2 & is_small1 & is_long & is_red
    cond_es_gap = (pd.concat([c1, o1], axis=1).min(axis=1) > c2)
    cond_es_close = c < ((o2 + c2) / 2)
    df["Pattern_Evening_Star"] = cond_es & cond_es_gap & cond_es_close
    
    # Three White Soldiers (Strong Bullish)
    # 3 Green, each closing higher
    df["Pattern_3_Soldiers"] = is_green & is_green1 & is_green2 & (c > c1) & (c1 > c2) & is_long & is_long1
    
    # Three Black Crows (Strong Bearish)
    # 3 Red, each closing lower
    df["Pattern_3_Crows"] = is_red & is_red1 & is_red2 & (c < c1) & (c1 < c2) & is_long & is_long1

    return df

def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    
    # --- EMA ---
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    # --- Bollinger Bands (20, 2.5) ---
    bb_mid = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_Mid"] = bb_mid
    df["BB_Upper"] = bb_mid + (2.5 * bb_std)
    df["BB_Lower"] = bb_mid - (2.5 * bb_std)

    # --- MACD ---
    # Standard: Fast=12, Slow=26, Signal=9
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # --- SuperTrend ---
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    
    # Calculate TR (True Range)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR
    # Period=10 (Optimized for TQQQ)
    df["ATR"] = df["TR"].ewm(alpha=1/10, adjust=False).mean()

    # Basic Bands
    # Multiplier=2.5 (Optimized for TQQQ)
    multiplier = 2.5
    basic_upper = (high + low) / 2 + multiplier * df["ATR"]
    basic_lower = (high + low) / 2 - multiplier * df["ATR"]
    
    # Final Bands & SuperTrend
    # We need to iterate because Final Bands depend on previous Final Bands
    final_upper = [0.0] * len(df)
    final_lower = [0.0] * len(df)
    supertrend = [0.0] * len(df)
    
    # 1 = Uptrend (Green), -1 = Downtrend (Red) 
    trend = [1] * len(df) 
    
    close_vals = close.values
    bu_vals = basic_upper.values
    bl_vals = basic_lower.values
    
    for i in range(1, len(df)):
        # Final Upper Band
        if bu_vals[i] < final_upper[i-1] or close_vals[i-1] > final_upper[i-1]:
            final_upper[i] = bu_vals[i]
        else:
            final_upper[i] = final_upper[i-1]
            
        # Final Lower Band
        if bl_vals[i] > final_lower[i-1] or close_vals[i-1] < final_lower[i-1]:
            final_lower[i] = bl_vals[i]
        else:
            final_lower[i] = final_lower[i-1]
            
        # Trend Direction
        if trend[i-1] == 1: # Previously Uptrend
            if close_vals[i] < final_lower[i]:
                trend[i] = -1 # Switch to Downtrend
            else:
                trend[i] = 1
        else: # Previously Downtrend
            if close_vals[i] > final_upper[i]:
                trend[i] = 1 # Switch to Uptrend
            else:
                trend[i] = -1
        
        # SuperTrend Value
        if trend[i] == 1:
            supertrend[i] = final_lower[i]
        else:
            supertrend[i] = final_upper[i]
            
    df["SuperTrend"] = pd.Series(supertrend, index=df.index, dtype="float64")
    df["SuperTrend_Dir"] = pd.Series(trend, index=df.index, dtype="int8")

    # --- ADX ---
    # Period=14
    # Already have TR in df["TR"]
    
    # Directional Movement
    up = high - high.shift()
    down = low.shift() - low
    
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    
    # +DM
    plus_dm[(up > down) & (up > 0)] = up[(up > down) & (up > 0)]
    
    # -DM
    minus_dm[(down > up) & (down > 0)] = down[(down > up) & (down > 0)]
    
    # Smoothed TR, +DM, -DM (Wilder's Smoothing)
    # First value is simple sum, subsequent are smoothed
    alpha = 1/14
    
    # We can use ewm with adjust=False for Wilder's if we set alpha=1/n
    tr_smooth = df["TR"].ewm(alpha=alpha, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()
    
    # DX
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # ADX
    df["ADX"] = dx.ewm(alpha=alpha, adjust=False).mean()
    df["Plus_DI"] = plus_di
    df["Minus_DI"] = minus_di

    # --- RSI ---
    # Period=14
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # --- Volume Context (20-day baseline) ---
    vol = df["Volume"].astype("float64")
    df["Volume_MA20"] = vol.rolling(window=20).mean()
    df["Volume_Ratio"] = vol / df["Volume_MA20"]

    price_up = df["Close"] > df["Close"].shift(1)
    adx_falling = df["ADX"] < df["ADX"].shift(1)

    # Weak rally requires underpowered volume and weakening trend strength.
    df["Volume_Weak_Rally"] = price_up & (vol < (df["Volume_MA20"] * 0.8)) & adx_falling

    # Graded warning levels for weak-rally context.
    df["Volume_Weak_Caution"] = price_up & adx_falling & (vol < (df["Volume_MA20"] * 0.9))
    df["Volume_Weak_Warning"] = price_up & adx_falling & (vol < (df["Volume_MA20"] * 0.7))
    df["Volume_Weak_Risk"] = price_up & adx_falling & (vol < (df["Volume_MA20"] * 0.5))

    # General abnormal volume spike.
    df["Volume_Climax"] = vol > (df["Volume_MA20"] * 2.5)
    # Blow-off top condition.
    daily_return = (df["Close"] / df["Close"].shift(1)) - 1.0
    df["Volume_Blowoff_Top"] = (daily_return > 0.03) & (vol > (df["Volume_MA20"] * 2.0)) & (df["RSI"] > 80)

    up_candle = df["Close"] > df["Open"]
    down_candle = df["Close"] < df["Open"]
    df["Volume_Accumulation_Day"] = up_candle & (vol > df["Volume_MA20"])
    df["Volume_Distribution_Day"] = down_candle & (vol > df["Volume_MA20"])
    df["Volume_Accumulation_10d"] = df["Volume_Accumulation_Day"].rolling(10).sum()
    df["Volume_Distribution_10d"] = df["Volume_Distribution_Day"].rolling(10).sum()

    # --- New Indicators for TQQQ ---
    # SMA 200 (Long-term trend filter)
    df["SMA200"] = df["Close"].rolling(window=200).mean()
    
    # RSI 2 (Short-term Mean Reversion)
    gain2 = (delta.where(delta > 0, 0)).ewm(alpha=1/2, adjust=False).mean()
    loss2 = (-delta.where(delta < 0, 0)).ewm(alpha=1/2, adjust=False).mean()
    rs2 = gain2 / loss2
    df["RSI_2"] = 100 - (100 / (1 + rs2))
    
    # --- Pattern Recognition ---
    df = identify_patterns(df)

    print(f"Indicators calculated. Rows: {len(df)}")
    return df
def calculate_signals_variant(df: pd.DataFrame, strategy_type: str = "conservative") -> pd.DataFrame:
    """
    Calculates Entry/Exit signals based on the selected strategy type.
    Returns the dataframe with 'Buy_Signal' and 'Sell_Signal' columns.
    """
    # Common Indicators
    st_dir = df["SuperTrend_Dir"]
    st_dir_prev = st_dir.shift(1)
    
    macd_line = df["MACD"]
    signal_line = df["MACD_Signal"]
    
    ema20 = df["EMA20"]
    ema50 = df["EMA50"]
    
    adx = df["ADX"]
    rsi = df["RSI"]
    
    # Initialize Signals
    buy_signal = pd.Series(False, index=df.index)
    sell_signal = pd.Series(False, index=df.index)
    
    # --- STRATEGY LOGIC ---
    
    if strategy_type == "ema_cross":
        # Buy: EMA20 crosses above EMA50
        # Sell: EMA20 crosses below EMA50
        buy_signal = (ema20 > ema50) & (ema20.shift(1) <= ema50.shift(1))
        sell_signal = (ema20 < ema50) & (ema20.shift(1) >= ema50.shift(1))
        
    elif strategy_type == "supertrend":
        # Buy: SuperTrend Flip Bullish
        # Sell: SuperTrend Flip Bearish
        buy_signal = (st_dir == 1) & (st_dir_prev == -1)
        sell_signal = (st_dir == -1) & (st_dir_prev == 1)
        
    elif strategy_type == "macd":
        # Buy: MACD Line crosses above Signal Line
        # Sell: MACD Line crosses below Signal Line
        buy_signal = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        sell_signal = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        
    elif strategy_type == "rsi_mean_reversion":
        # Buy: RSI < 30 (Oversold)
        # Sell: RSI > 70 (Overbought)
        buy_signal = (rsi < 30)
        sell_signal = (rsi > 70)
        
    elif strategy_type == "rsi_trend":
        # Buy: RSI > 50
        # Sell: RSI < 50
        buy_signal = (rsi > 50) & (rsi.shift(1) <= 50)
        sell_signal = (rsi < 50) & (rsi.shift(1) >= 50)
        
    elif strategy_type == "aggressive":
        # SuperTrend + MACD (No ADX, No EMA)
        st_flip_bull = (st_dir == 1) & (st_dir_prev == -1)
        st_flip_bear = (st_dir == -1) & (st_dir_prev == 1)
        macd_bull = (macd_line > signal_line)
        macd_bear = (macd_line < signal_line)
        
        buy_signal = st_flip_bull & macd_bull
        sell_signal = st_flip_bear & macd_bear
        
    elif strategy_type == "conservative":
        # Current Logic: ST Flip + MACD(Zero) + EMA + ADX>=20
        st_flip_bull = (st_dir == 1) & (st_dir_prev == -1)
        st_flip_bear = (st_dir == -1) & (st_dir_prev == 1)
        macd_bull = (macd_line > signal_line) & (macd_line > 0)
        macd_bear = (macd_line < signal_line) & (macd_line < 0)
        trend_bull = (ema20 > ema50)
        trend_bear = (ema20 < ema50)
        adx_filter = (adx >= 20)
        
        buy_signal = st_flip_bull & macd_bull & trend_bull & adx_filter
        sell_signal = st_flip_bear & macd_bear & trend_bear & adx_filter
        
    elif strategy_type == "hybrid_asymmetric":
        # Asymmetric: Aggressive Entry + Fast Exit
        # BUY: SuperTrend + MACD (strict)
        # SELL: EMA20/50 Death Cross (fast)
        st_flip_bull = (st_dir == 1) & (st_dir_prev == -1)
        macd_bull = (macd_line > signal_line)
        
        buy_signal = st_flip_bull & macd_bull
        
        # Sell on EMA death cross
        sell_signal = (ema20 < ema50) & (ema20.shift(1) >= ema50.shift(1))
        
    elif strategy_type == "rsi_2_mean_reversion":
        # Larry Connors Strategy for ETFs
        # Buy: Price is above SMA200 (Long term uptrend) AND RSI(2) dips below 10
        # Sell: RSI(2) rises above 70
        sma200 = df["SMA200"]
        rsi2 = df["RSI_2"]
        
        buy_signal = (df["Close"] > sma200) & (rsi2 < 10)
        sell_signal = (rsi2 > 70)
        
    elif strategy_type == "volatility_breakout":
        # Modified VBO for Daily Candles
        # Buy: Strong Momentum (Today's Close > Open + 0.5 * Yesterday's Range)
        # Sell: Momentum lost (Today's Close < Open)
        prev_range = (df["High"].shift(1) - df["Low"].shift(1))
        k = 0.5
        target = df["Open"] + (prev_range * k)
        
        buy_signal = (df["Close"] > target)
        sell_signal = (df["Close"] < df["Open"])

    else:
        print(f"Unknown strategy: {strategy_type}")

    df["Buy_Signal"] = buy_signal
    df["Sell_Signal"] = sell_signal
    return df

def calculate_signals(df: pd.DataFrame) -> pd.DataFrame:
    # Default to conservative for backward compatibility
    return calculate_signals_variant(df, "conservative")

def detect_trigger(data: pd.DataFrame) -> TriggerResult:
    if len(data) < 2:
        return TriggerResult("Insufficient data", "#999999")

    last = data.iloc[-1]
    prev = data.iloc[-2]
    
    # Extract values
    st_dir = last["SuperTrend_Dir"] # 1 or -1
    st_dir_prev = prev["SuperTrend_Dir"]
    
    date_str = last.name.strftime("%Y-%m-%d")

    # Check Patterns
    p_bull_eng = last["Pattern_Bull_Engulfing"]
    p_hammer = last["Pattern_Hammer"]
    p_morning = last["Pattern_Morning_Star"]
    p_soldiers = last["Pattern_3_Soldiers"]
    p_bull_harami = last["Pattern_Bull_Harami"]
    p_piercing = last["Pattern_Piercing"]
    
    p_bear_eng = last["Pattern_Bear_Engulfing"]
    p_shooting = last["Pattern_Shooting_Star"]
    p_evening = last["Pattern_Evening_Star"]
    p_crows = last["Pattern_3_Crows"]
    p_bear_harami = last["Pattern_Bear_Harami"]
    p_dark_cloud = last["Pattern_Dark_Cloud"]
    
    pat_str = ""
    # Bullish
    if p_morning: pat_str += " [Morning Star]"
    if p_soldiers: pat_str += " [3 Soldiers]"
    if p_bull_eng: pat_str += " [Bull Engulfing]"
    if p_piercing: pat_str += " [Piercing]"
    if p_bull_harami: pat_str += " [Bull Harami]"
    if p_hammer: pat_str += " [Hammer]"
    
    # Bearish
    if p_evening: pat_str += " [Evening Star]"
    if p_crows: pat_str += " [3 Crows]"
    if p_bear_eng: pat_str += " [Bear Engulfing]"
    if p_dark_cloud: pat_str += " [Dark Cloud]"
    if p_bear_harami: pat_str += " [Bear Harami]"
    if p_shooting: pat_str += " [Shooting Star]"

    # --- Pattern Energy (Sentiment Analysis) ---
    bull_cols = ["Pattern_Bull_Engulfing", "Pattern_Hammer", "Pattern_Morning_Star", 
                 "Pattern_3_Soldiers", "Pattern_Bull_Harami", "Pattern_Piercing"]
    bear_cols = ["Pattern_Bear_Engulfing", "Pattern_Shooting_Star", "Pattern_Evening_Star", 
                 "Pattern_3_Crows", "Pattern_Bear_Harami", "Pattern_Dark_Cloud"]
    
    recent_20 = data.tail(20)
    
    # Check if columns exist before summing
    existing_bull = [c for c in bull_cols if c in data.columns]
    existing_bear = [c for c in bear_cols if c in data.columns]
    
    bull_count = int(recent_20[existing_bull].sum().sum())
    bear_count = int(recent_20[existing_bear].sum().sum())
    
    sentiment = "Mixed ⚖️"
    if bull_count > bear_count * 1.5 and bull_count > 2:
        sentiment = "Bulls Dominate 🐂"
    elif bear_count > bull_count * 1.5 and bear_count > 2:
        sentiment = "Bears Dominate 🐻"
    
    energy_msg = f" | 20d Pattern Energy: {bull_count} Bull vs {bear_count} Bear ({sentiment})"

    # --- Volume Context Message ---
    volume_notes = []
    if "Volume_Weak_Risk" in data.columns and bool(last["Volume_Weak_Risk"]):
        volume_notes.append("Weak Rally Risk")
    elif "Volume_Weak_Warning" in data.columns and bool(last["Volume_Weak_Warning"]):
        volume_notes.append("Weak Rally Warning")
    elif "Volume_Weak_Caution" in data.columns and bool(last["Volume_Weak_Caution"]):
        volume_notes.append("Weak Rally Caution")

    if "Volume_Blowoff_Top" in data.columns and bool(last["Volume_Blowoff_Top"]):
        volume_notes.append("Blow-off Top")
    elif "Volume_Climax" in data.columns and bool(last["Volume_Climax"]):
        volume_notes.append("Climax Volume")
    if "Volume_Accumulation_10d" in data.columns and "Volume_Distribution_10d" in data.columns:
        accum_10 = float(last["Volume_Accumulation_10d"])
        dist_10 = float(last["Volume_Distribution_10d"])
        if accum_10 >= dist_10 + 2:
            volume_notes.append("Accumulation Bias")
        elif dist_10 >= accum_10 + 2:
            volume_notes.append("Distribution Bias")

    volume_msg = ""
    if volume_notes:
        volume_msg = " | Vol: " + ", ".join(volume_notes)

    # 1. BULLISH SCENARIO
    if st_dir == 1:
        if st_dir_prev == -1:
            msg = f"★ BUY SIGNAL (Flip) on {date_str}{pat_str}{energy_msg}{volume_msg}"
            color = "#00cc00" # Bright Green
        else:
            msg = f"HOLD LONG (Trend is Bullish) since {date_str}{pat_str}{energy_msg}{volume_msg}"
            color = "#2ca02c" # Standard Green
        return TriggerResult(msg, color)

    # 2. BEARISH SCENARIO
    if st_dir == -1:
        if st_dir_prev == 1:
            msg = f"★ SELL SIGNAL (Flip) on {date_str}{pat_str}{energy_msg}{volume_msg}"
            color = "#ff0000" # Bright Red
        else:
            msg = f"STAY CASH (Trend is Bearish) since {date_str}{pat_str}{energy_msg}{volume_msg}"
            color = "#ff7f0e" # Orange
        return TriggerResult(msg, color)

    return TriggerResult("Neutral", "#999999")
