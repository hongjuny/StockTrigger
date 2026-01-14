import sys
import numpy as np
from dataclasses import dataclass
from typing import Optional
import strategy

import mplfinance as mpf
import pandas as pd
import yfinance as yf
from matplotlib import dates as mdates
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


yf.set_tz_cache_location("./.yfinance_cache")

LOOKBACK_OPTIONS = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
}

LOOKBACK_DAYS = {
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
}

# Add buffer days for EMA calculation (need at least 50 candles before display window)
FETCH_BUFFER_DAYS = 100

INTERVAL_OPTIONS = {
    "Daily": "1d",
    "Weekly": "1wk",
}





class DataWorker(QThread):
    data_fetched = pyqtSignal(object, object, object)  # display_df, full_df, trigger_result
    error_occurred = pyqtSignal(str)

    def __init__(self, symbol: str, period: str, interval: str, lookback_days: int):
        super().__init__()
        self.symbol = symbol
        self.period = period
        self.interval = interval
        self.lookback_days = lookback_days

    def run(self):
        try:
            # Calculate fetch window
            display_days = self.lookback_days
            total_days = display_days + FETCH_BUFFER_DAYS
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=total_days)

            print(f"Worker: Fetching {self.symbol} from {start_date.date()} to {end_date.date()}")

            data = yf.download(
                tickers=self.symbol,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval=self.interval,
                auto_adjust=False,
                progress=False,
            )

            if data.empty:
                raise ValueError("No data returned. Try a different symbol or interval.")

            # Flatten MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()
            
            # Fix timezone if present
            if hasattr(data.index, "tz_localize"):
                try:
                    data.index = data.index.tz_localize(None)
                except TypeError:
                    pass

            # Strategy & Indicators
            ema_df = strategy.add_indicators(data)
            ema_df = strategy.calculate_signals_variant(ema_df, "supertrend")
            
            # Filter Display Window
            cutoff = ema_df.index.max() - pd.Timedelta(days=display_days)
            display_df = ema_df[ema_df.index >= cutoff]
            if display_df.empty:
                display_df = ema_df
            
            trigger = strategy.detect_trigger(ema_df)
            
            self.data_fetched.emit(display_df, ema_df, trigger)

        except Exception as e:
            self.error_occurred.emit(str(e))


class StockTriggerApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EMA Trigger Dashboard")
        self.resize(1100, 700)

        self._init_ui()

    def _init_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout()
        central_widget.setLayout(root_layout)

        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(12)

        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("e.g., TQQQ, SQQQ, AAPL")
        self.symbol_input.setText("TQQQ")
        self.symbol_input.setMaximumWidth(180)
        self.symbol_input.returnPressed.connect(self.handle_fetch)

        self.lookback_combo = QComboBox()
        for label, value in LOOKBACK_OPTIONS.items():
            self.lookback_combo.addItem(label, value)
        self.lookback_combo.setCurrentIndex(3)  # default 1 year
        self.lookback_combo.currentIndexChanged.connect(self.handle_fetch)

        self.interval_combo = QComboBox()
        for label, value in INTERVAL_OPTIONS.items():
            self.interval_combo.addItem(label, value)
        self.interval_combo.currentIndexChanged.connect(self.handle_fetch)

        self.fetch_button = QPushButton("Fetch Data")
        self.fetch_button.setMinimumHeight(34)
        self.fetch_button.clicked.connect(self.handle_fetch)

        # Indicator visibility checkboxes
        self.show_volume_cb = QCheckBox("Volume")
        self.show_volume_cb.setChecked(True)
        self.show_volume_cb.stateChanged.connect(self.update_chart_layout)
        
        self.show_macd_cb = QCheckBox("MACD")
        self.show_macd_cb.setChecked(True)
        self.show_macd_cb.stateChanged.connect(self.update_chart_layout)
        
        self.show_adx_cb = QCheckBox("ADX/DMI")
        self.show_adx_cb.setChecked(True)
        self.show_adx_cb.stateChanged.connect(self.update_chart_layout)

        controls_layout.addWidget(QLabel("Symbol"))
        controls_layout.addWidget(self.symbol_input)
        controls_layout.addWidget(QLabel("Lookback"))
        controls_layout.addWidget(self.lookback_combo)
        controls_layout.addWidget(QLabel("Interval"))
        controls_layout.addWidget(self.interval_combo)
        controls_layout.addWidget(self.fetch_button)
        controls_layout.addWidget(QLabel("|"))
        controls_layout.addWidget(QLabel("Show:"))
        controls_layout.addWidget(self.show_volume_cb)
        controls_layout.addWidget(self.show_macd_cb)
        controls_layout.addWidget(self.show_adx_cb)
        controls_layout.addStretch(1)

        root_layout.addLayout(controls_layout)

        info_layout = QHBoxLayout()
        self.trigger_label = QLabel("Trigger: --")
        self.trigger_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.trigger_label.setStyleSheet("font-weight: bold; font-size: 16px;")

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.status_label.setStyleSheet("color: #555;")

        info_layout.addWidget(self.trigger_label)
        info_layout.addWidget(self.status_label)
        root_layout.addLayout(info_layout)

        self.figure = Figure(figsize=(10, 10), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        
        # Initial Chart Setup
        self.current_data = None
        self.update_chart_layout()

        root_layout.addWidget(self.canvas, stretch=1)

    def update_chart_layout(self) -> None:
        """Recreate chart layout based on checkbox states."""
        self.figure.clear()
        
        # Determine which panels to show
        show_volume = self.show_volume_cb.isChecked()
        show_macd = self.show_macd_cb.isChecked()
        show_adx = self.show_adx_cb.isChecked()
        
        # Build height ratios dynamically
        num_panels = 1  # Always have price
        height_ratios = [3]  # Price panel
        
        if show_volume:
            num_panels += 1
            height_ratios.append(1)
        if show_macd:
            num_panels += 1
            height_ratios.append(1)
        if show_adx:
            num_panels += 1
            height_ratios.append(1)
        
        # Create grid with dynamic panels
        grid = self.figure.add_gridspec(num_panels, 1, height_ratios=height_ratios)
        
        # Always create price axis
        self.price_ax = self.figure.add_subplot(grid[0, 0])
        
        # Create other axes based on checkboxes
        panel_idx = 1
        
        if show_volume:
            self.volume_ax = self.figure.add_subplot(grid[panel_idx, 0], sharex=self.price_ax)
            panel_idx += 1
        else:
            self.volume_ax = None
            
        if show_macd:
            self.macd_ax = self.figure.add_subplot(grid[panel_idx, 0], sharex=self.price_ax)
            panel_idx += 1
        else:
            self.macd_ax = None
            
        if show_adx:
            self.adx_ax = self.figure.add_subplot(grid[panel_idx, 0], sharex=self.price_ax)
        else:
            self.adx_ax = None
        
        # Re-plot if we have data
        if self.current_data is not None:
            self.update_plot(self.current_data)

    def handle_fetch(self) -> None:
        symbol = self.symbol_input.text().strip().upper()
        if not symbol:
            QMessageBox.warning(self, "Input Required", "Please enter a ticker symbol.")
            return

        display_period = self.lookback_combo.currentData() # e.g. "1y"
        lookback_days = LOOKBACK_DAYS.get(display_period, 365)
        interval = self.interval_combo.currentData()

        self.status_label.setText("Fetching data (background)...")
        self.fetch_button.setEnabled(False)
        
        # Create and start worker
        self.worker = DataWorker(symbol, display_period, interval, lookback_days)
        self.worker.data_fetched.connect(self.on_data_success)
        self.worker.error_occurred.connect(self.on_data_error)
        self.worker.finished.connect(lambda: self.fetch_button.setEnabled(True))
        self.worker.start()

    def on_data_success(self, display_df, full_df, trigger):
        self.current_data = display_df
        self.update_plot(display_df)
        
        self.trigger_label.setText(f"Trigger: {trigger.message}")
        self.trigger_label.setStyleSheet(
            f"font-weight: bold; font-size: 16px; color: {trigger.color};"
        )
        self.status_label.setText(
            f"Showing {len(display_df)} candles ({self.lookback_combo.currentText()}, {self.interval_combo.currentText().lower()})"
        )

    def on_data_error(self, error_msg):
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Fetch Error", error_msg)



    def update_plot(self, data: pd.DataFrame) -> None:
        self.price_ax.clear()
        if self.volume_ax: self.volume_ax.clear()
        if self.macd_ax: self.macd_ax.clear()
        if self.adx_ax: self.adx_ax.clear()

        price_df = data.copy()
        x_indices = np.arange(len(price_df))
        
        # --- SuperTrend Lines ---
        st_up = price_df["SuperTrend"].copy()
        st_down = price_df["SuperTrend"].copy()
        st_up[price_df["SuperTrend_Dir"] == -1] = np.nan
        st_down[price_df["SuperTrend_Dir"] == 1] = np.nan

        # --- Buy/Sell Signals ---
        buy_signals = pd.Series(np.nan, index=price_df.index)
        sell_signals = pd.Series(np.nan, index=price_df.index)
        st_dir = price_df["SuperTrend_Dir"]
        st_dir_prev = st_dir.shift(1)
        
        buy_mask = (st_dir == 1) & (st_dir_prev == -1)
        sell_mask = (st_dir == -1) & (st_dir_prev == 1)
        
        buy_signals[buy_mask] = price_df.loc[buy_mask, "Low"] * 0.98
        sell_signals[sell_mask] = price_df.loc[sell_mask, "High"] * 1.02

        # --- Pattern Markers ---
        bull_cols = ["Pattern_Bull_Engulfing", "Pattern_Hammer", "Pattern_Morning_Star", 
                     "Pattern_3_Soldiers", "Pattern_Bull_Harami", "Pattern_Piercing"]
        bull_pat_mask = price_df[bull_cols[0]].copy()
        for col in bull_cols[1:]:
            if col in price_df.columns: bull_pat_mask |= price_df[col]
        
        bull_pat_signals = pd.Series(np.nan, index=price_df.index)
        bull_pat_signals[bull_pat_mask] = price_df.loc[bull_pat_mask, "Low"] * 0.96

        bear_cols = ["Pattern_Bear_Engulfing", "Pattern_Shooting_Star", "Pattern_Evening_Star", 
                     "Pattern_3_Crows", "Pattern_Bear_Harami", "Pattern_Dark_Cloud"]
        bear_pat_mask = price_df[bear_cols[0]].copy()
        for col in bear_cols[1:]:
            if col in price_df.columns: bear_pat_mask |= price_df[col]

        bear_pat_signals = pd.Series(np.nan, index=price_df.index)
        bear_pat_signals[bear_pat_mask] = price_df.loc[bear_pat_mask, "High"] * 1.04

        # Addplots for MAIN axis only
        price_addplots = [
            mpf.make_addplot(price_df["EMA20"], ax=self.price_ax, color="orange", width=0.8),
            mpf.make_addplot(price_df["EMA50"], ax=self.price_ax, color="blue", width=0.8),
            mpf.make_addplot(st_up, ax=self.price_ax, color="green", width=2.0),
            mpf.make_addplot(st_down, ax=self.price_ax, color="red", width=2.0),
            mpf.make_addplot(buy_signals, ax=self.price_ax, type='scatter', markersize=140, marker='^', color='green'),
            mpf.make_addplot(sell_signals, ax=self.price_ax, type='scatter', markersize=140, marker='v', color='red'),
            mpf.make_addplot(bull_pat_signals, ax=self.price_ax, type='scatter', markersize=100, marker='*', color='#00FF00'),
            mpf.make_addplot(bear_pat_signals, ax=self.price_ax, type='scatter', markersize=100, marker='*', color='#FF0000'),
        ]

        # 1. Plot Candle and Main Indicators
        mpf.plot(
            price_df,
            type="candle",
            ax=self.price_ax,
            volume=self.volume_ax if self.volume_ax else False,
            addplot=price_addplots,
            style="yahoo",
            show_nontrading=False,
            datetime_format="%Y-%m-%d",
        )

        # 2. Manually Plot MACD if axis exists
        if self.macd_ax:
            self.macd_ax.plot(x_indices, price_df["MACD"], color="black", lw=1.0, label="MACD")
            self.macd_ax.plot(x_indices, price_df["MACD_Signal"], color="red", lw=1.0, label="Signal")
            # Histogram
            hist = price_df["MACD_Hist"]
            colors = ["green" if val > 0 else "red" for val in hist]
            self.macd_ax.bar(x_indices, hist, color=colors, alpha=0.3, width=0.7)
            self.macd_ax.legend(loc="upper left", fontsize=8, frameon=False)
            self.macd_ax.set_ylabel("MACD")

        # 3. Manually Plot ADX if axis exists
        if self.adx_ax:
            self.adx_ax.plot(x_indices, price_df["ADX"], color="black", lw=1.2, label="ADX")
            self.adx_ax.plot(x_indices, price_df["Plus_DI"], color="green", lw=0.8, ls=":", label="+DI")
            self.adx_ax.plot(x_indices, price_df["Minus_DI"], color="red", lw=0.8, ls=":", label="-DI")
            self.adx_ax.axhline(25, color="gray", lw=0.8, ls="--", alpha=0.5)
            self.adx_ax.legend(loc="upper left", fontsize=8, frameon=False)
            self.adx_ax.set_ylabel("ADX/DMI")

        # 4. Final Formatting for all visible axes
        all_axes = [ax for ax in [self.price_ax, self.volume_ax, self.macd_ax, self.adx_ax] if ax is not None]
        for ax in all_axes:
            ax.set_xlim(-0.5, len(price_df) - 0.5)
            ax.yaxis.set_label_position("left")
            ax.yaxis.tick_left()
            ax.grid(True, alpha=0.15)
            # Hide x-labels except for bottom one
            if ax != all_axes[-1]:
                plt.setp(ax.get_xticklabels(), visible=False)
        
        # Bottom axis date formatting (using mplfinance index mapping)
        bottom_ax = all_axes[-1]
        # We need to set the labels manually since we might have hidden some
        tick_indices = np.linspace(0, len(price_df)-1, 8).astype(int)
        bottom_ax.set_xticks(tick_indices)
        bottom_ax.set_xticklabels(
            [price_df.index[i].strftime("%Y-%m-%d") for i in tick_indices],
            rotation=0, fontsize=7
        )

        # Price Legend
        self.price_ax.legend(
            [
                Line2D([0], [0], color="orange", lw=1.0),
                Line2D([0], [0], color="blue", lw=1.0),
                Line2D([0], [0], color="green", lw=2.0), 
                Line2D([0], [0], color="red", lw=2.0),
                Line2D([0], [0], marker='*', color='#00FF00', linestyle='None', markersize=10),
                Line2D([0], [0], marker='*', color='#FF0000', linestyle='None', markersize=10),
            ],
            ["EMA20", "EMA50", "ST Bull", "ST Bear", "Bull Pattern", "Bear Pattern"],
            loc="upper left", fontsize=9, frameon=True
        )

        # --- Annotate Patterns (Educational Mode) ---
        # Iterate to add text labels for patterns
        # We use a slight offset to avoid overlapping with markers
        
        # Define priority patterns to label (short names)
        pat_map = {
            "Pattern_Morning_Star": "M-Star",
            "Pattern_Evening_Star": "E-Star",
            "Pattern_3_Soldiers": "3-Sold",
            "Pattern_3_Crows": "3-Crows",
            "Pattern_Bull_Engulfing": "Engulf",
            "Pattern_Bear_Engulfing": "Engulf",
            "Pattern_Hammer": "Hammer",
            "Pattern_Shooting_Star": "Star",
            "Pattern_Piercing": "Pierce",
            "Pattern_Dark_Cloud": "Cloud",
            "Pattern_Bull_Harami": "Harami",
            "Pattern_Bear_Harami": "Harami"
        }
        
        lows = price_df["Low"].values
        highs = price_df["High"].values
        
        for i in range(len(price_df)):
            # Bullish Labels
            bull_label = []
            for col, name in pat_map.items():
                if col in price_df.columns and price_df[col].iloc[i]:
                    if "Bull" in col or "Morning" in col or "Soldiers" in col or "Hammer" in col or "Piercing" in col:
                        bull_label.append(name)
            
            if bull_label:
                # Take the most significant one (or join them)
                lbl = "/".join(bull_label[:2]) # Max 2 patterns to keep short
                self.price_ax.text(i, lows[i] * 0.94, lbl, color='#00CC00', fontsize=7, ha='center', va='top', rotation=0, fontweight='bold')

            # Bearish Labels
            bear_label = []
            for col, name in pat_map.items():
                if col in price_df.columns and price_df[col].iloc[i]:
                    if "Bear" in col or "Evening" in col or "Crows" in col or "Shooting" in col or "Dark" in col:
                        bear_label.append(name)
            
            if bear_label:
                lbl = "/".join(bear_label[:2])
                self.price_ax.text(i, highs[i] * 1.05, lbl, color='#CC0000', fontsize=7, ha='center', va='bottom', rotation=0, fontweight='bold')

        self.canvas.draw_idle()


def main() -> None:
    app = QApplication(sys.argv)
    window = StockTriggerApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
