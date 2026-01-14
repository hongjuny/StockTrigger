# Stock Trigger Dashboard (TQQQ Optimized)

A PyQt5-based desktop application designed for **TQQQ (ProShares UltraPro QQQ)** trading analysis. It visualizes market trends using **SuperTrend**, identifies **12 key candlestick patterns**, and calculates **Pattern Energy** to gauge market sentiment.

![Dashboard Preview](https://via.placeholder.com/800x400?text=Stock+Trigger+Dashboard) *(Replace with actual screenshot)*

## Backtest Performance (TQQQ)

We conducted a 10-year historical backtest (2014–2024) to evaluate the **SuperTrend (10, 2.5)** strategy against a simple Buy & Hold approach for TQQQ.

| Metric | SuperTrend (Optimized) | Buy & Hold |
| :--- | :--- | :--- |
| **Total Return** | **~1,028%** | ~2,721% |
| **Max Drawdown (MDD)** | **-42.02%** | **-81.75%** |
| **Recovery Speed** | Fast | Very Slow |
| **Trading Style** | Defensive Trend Following | Passive (High Risk) |

### Why this strategy?
While Buy & Hold shows a higher total return on paper, an **82% drawdown** is psychologically and financially devastating for most investors. Our optimized SuperTrend strategy captures significant upside while **cutting the maximum loss in half**, ensuring you stay in the game during major market crashes like COVID-19 (2020) or the 2022 Bear Market.

---

## Trading Philosophy

> **"Don't be a hero, be a survivor."**

TQQQ is a triple-leveraged ETF, meaning **Volatility Decay** and **Drawdowns** are your biggest enemies. This dashboard is built on two pillars:
1.  **Confluence:** We don't just look at one line. We wait for the Trend (SuperTrend) to align with Candle Patterns and Momentum (MACD/ADX).
2.  **Objectivity:** Emotions kill trades. The dashboard provides clear, math-based signals to remove "gut feelings" from your execution.

---

## How to Use the Dashboard

### 1. Identify the Primary Trend
Look at the **SuperTrend Line**. 
*   **Green:** The wind is at your back. Stay long.
*   **Red:** The storm has arrived. Stay in cash.

### 2. Look for Pattern Confirmation (Stars ★)
*   If a **Green Star (★)** appears during a Green SuperTrend, it's a **strong continuation/buy signal**.
*   If a **Red Star (★)** appears, even in a Green trend, it's a **warning** to tighten your stop-losses.

### 3. Check the "Pattern Energy"
The 상단 label shows the **20-day Pattern Energy**. 
*   If it says `Bulls Dominate 🐂`, the underlying buying pressure is strong despite any short-term dips.

---

## Project Layout

*   `main.py`: The GUI application. Launch this to see the dashboard.
*   `strategy.py`: The "Brain". Contains all technical calculations and pattern recognition logic.
*   `backtest.py`: Historical simulation engine. Verify the numbers yourself.
*   `grid_search.py`: Optimization tool. Use this to find the best settings for other stocks (e.g., SOXL, NVDA).
*   `requirements.txt`: Necessary Python packages.

---

## Technical Details

### 12 Candlestick Patterns Included:
*   **Reversal (Bullish):** Morning Star, Bullish Engulfing, Hammer, Piercing Line, Bullish Harami, Three White Soldiers.
*   **Reversal (Bearish):** Evening Star, Bearish Engulfing, Shooting Star, Dark Cloud Cover, Bearish Harami, Three Black Crows.

### Optimized Indicators:
*   **SuperTrend:** Period 10, Multiplier 2.5
*   **EMA:** 20 (Fast), 50 (Slow)
*   **MACD:** 12, 26, 9
*   **ADX:** 14-period Trend Strength filter.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/stock-trigger.git
    cd stock-trigger
    ```

2.  **Set up a virtual environment (Recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Mac/Linux
    # .venv\Scripts\activate   # Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the application:**
    ```bash
    python main.py
    ```

2.  **Dashboard Controls:**
    *   **Symbol:** Default is `TQQQ`. Can be changed to `AAPL`, `NVDA`, `SOXL`, etc.
    *   **Lookback:** 1 Month to 1 Year.
    *   **Interval:** Daily or Weekly.
    *   **Checkboxes:** Toggle Volume, MACD, ADX panels.

3.  **Interpreting Signals:**
    *   **Green Line/Arrow:** Bullish Trend (Hold Long).
    *   **Red Line/Arrow:** Bearish Trend (Stay Cash).
    *   **Green Star (★):** Bullish Candle Pattern detected.
    *   **Red Star (★):** Bearish Candle Pattern detected.
    *   **Trigger Label:** Shows the current action (e.g., `★ BUY SIGNAL`) and specific patterns (e.g., `[Morning Star]`).

## Backtesting & Grid Search

*   **`backtest.py`**: Run a historical simulation of the strategy.
    ```bash
    python backtest.py
    ```
*   **`grid_search.py`**: Find optimal SuperTrend parameters (Period/Multiplier) for a specific stock.
    ```bash
    python grid_search.py
    ```

## Dependencies
*   Python 3.8+
*   PyQt5
*   pandas, numpy
*   mplfinance, matplotlib
*   yfinance

## License
MIT License