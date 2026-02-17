import sys
import json
import numpy as np
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import strategy

import mplfinance as mpf
import pandas as pd
import yfinance as yf
from matplotlib import dates as mdates
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
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

# Persist recent symbols and last selected symbol across app restarts.
UI_STATE_FILE = Path(".trigger_ui_state.json")
MAX_RECENT_SYMBOLS = 10

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
            # Keep chart buy/sell markers aligned with the backtest baseline logic.
            ema_df = strategy.calculate_signals_variant(ema_df, "conservative")
            
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
        self.resize(1200, 1200)
        self.recent_symbols = []
        self.last_symbol = "TQQQ"
        self._load_ui_state()

        self._init_ui()
        # Auto-load the most recent symbol once the window event loop starts.
        QTimer.singleShot(0, self.handle_fetch)

    def _init_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout()
        central_widget.setLayout(root_layout)

        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(12)

        self.symbol_input = QComboBox()
        self.symbol_input.setEditable(True)
        self.symbol_input.setInsertPolicy(QComboBox.NoInsert)
        self.symbol_input.setMaximumWidth(200)
        self.symbol_input.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.symbol_input.setToolTip("Type ticker or pick from recent symbols")
        self.symbol_input.lineEdit().setPlaceholderText("e.g., TQQQ, SQQQ, AAPL")
        self.symbol_input.activated.connect(lambda _: self.handle_fetch())
        self._refresh_symbol_combo()
        self.symbol_input.lineEdit().returnPressed.connect(self.handle_fetch)

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

        self.show_supertrend_cb = QCheckBox("SuperTrend")
        self.show_supertrend_cb.setChecked(True)
        self.show_supertrend_cb.stateChanged.connect(self.update_chart_layout)
        
        self.show_adx_cb = QCheckBox("ADX/DMI")
        self.show_adx_cb.setChecked(True)
        self.show_adx_cb.stateChanged.connect(self.update_chart_layout)

        self.show_rsi_cb = QCheckBox("RSI")
        self.show_rsi_cb.setChecked(True)
        self.show_rsi_cb.stateChanged.connect(self.update_chart_layout)

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
        controls_layout.addWidget(self.show_supertrend_cb)
        controls_layout.addWidget(self.show_macd_cb)
        controls_layout.addWidget(self.show_adx_cb)
        controls_layout.addWidget(self.show_rsi_cb)
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

        # Hover price display (no chart overlays)
        self._base_status_text = "Ready"
        self._hover_line = None
        self._hover_text = None
        self._hover_ylim = None
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)


    def update_chart_layout(self) -> None:
        """Recreate chart layout based on checkbox states."""
        self.figure.clear()
        
        # Determine which panels to show
        show_volume = self.show_volume_cb.isChecked()
        show_macd = self.show_macd_cb.isChecked()
        show_adx = self.show_adx_cb.isChecked()
        show_rsi = self.show_rsi_cb.isChecked()
        
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
        if show_rsi:
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
            panel_idx += 1
        else:
            self.adx_ax = None

        if show_rsi:
            self.rsi_ax = self.figure.add_subplot(grid[panel_idx, 0], sharex=self.price_ax)
        else:
            self.rsi_ax = None
        
        # Re-plot if we have data
        if self.current_data is not None:
            self.update_plot(self.current_data)

    def handle_fetch(self) -> None:
        symbol = self._get_symbol_text()
        if not symbol:
            QMessageBox.warning(self, "Input Required", "Please enter a ticker symbol.")
            return
        self._remember_symbol(symbol)

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
        self._base_status_text = self.status_label.text()

    def on_data_error(self, error_msg):
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Fetch Error", error_msg)



    def update_plot(self, data: pd.DataFrame) -> None:
        self.price_ax.clear()
        if self.volume_ax: self.volume_ax.clear()
        if self.macd_ax: self.macd_ax.clear()
        if self.adx_ax: self.adx_ax.clear()
        if self.rsi_ax: self.rsi_ax.clear()

        price_df = data.copy()
        x_indices = np.arange(len(price_df))
        show_supertrend = self.show_supertrend_cb.isChecked()
        
        # --- SuperTrend Lines ---
        st_up = price_df["SuperTrend"].copy()
        st_down = price_df["SuperTrend"].copy()
        st_up[price_df["SuperTrend_Dir"] == -1] = np.nan
        st_down[price_df["SuperTrend_Dir"] == 1] = np.nan

        # --- Buy/Sell Signals ---
        buy_signals = pd.Series(np.nan, index=price_df.index)
        sell_signals = pd.Series(np.nan, index=price_df.index)
        buy_mask = price_df["Buy_Signal"].fillna(False)
        sell_mask = price_df["Sell_Signal"].fillna(False)
        
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
            mpf.make_addplot(buy_signals, ax=self.price_ax, type='scatter', markersize=140, marker='^', color='green'),
            mpf.make_addplot(sell_signals, ax=self.price_ax, type='scatter', markersize=140, marker='v', color='red'),
            mpf.make_addplot(bull_pat_signals, ax=self.price_ax, type='scatter', markersize=100, marker='*', color='#00FF00'),
            mpf.make_addplot(bear_pat_signals, ax=self.price_ax, type='scatter', markersize=100, marker='*', color='#FF0000'),
        ]
        if show_supertrend:
            price_addplots.insert(2, mpf.make_addplot(st_up, ax=self.price_ax, color="green", width=2.0))
            price_addplots.insert(3, mpf.make_addplot(st_down, ax=self.price_ax, color="red", width=2.0))

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

        # Volume MA20 overlay on the volume panel.
        if self.volume_ax:
            vol_ma20 = price_df.get("Volume_MA20", price_df["Volume"].rolling(window=20).mean())
            self.volume_ax.plot(
                x_indices,
                vol_ma20,
                color="#ff8c00",
                lw=1.0,
                label="Vol MA20",
            )
            if "Volume_Climax" in price_df.columns:
                climax_mask = price_df["Volume_Climax"].fillna(False).to_numpy()
                if climax_mask.any():
                    self.volume_ax.bar(
                        x_indices[climax_mask],
                        price_df["Volume"].to_numpy()[climax_mask],
                        color="#5a189a",
                        alpha=0.45,
                        width=0.7,
                        label="Climax",
                    )
            if "Volume_Blowoff_Top" in price_df.columns:
                blowoff_mask = price_df["Volume_Blowoff_Top"].fillna(False).to_numpy()
                if blowoff_mask.any():
                    self.volume_ax.bar(
                        x_indices[blowoff_mask],
                        price_df["Volume"].to_numpy()[blowoff_mask],
                        color="#111111",
                        alpha=0.6,
                        width=0.7,
                        label="Blow-off",
                    )
            if "Volume_Weak_Caution" in price_df.columns:
                weak_caution = price_df["Volume_Weak_Caution"].fillna(False).to_numpy()
                if weak_caution.any():
                    self.volume_ax.scatter(
                        x_indices[weak_caution],
                        price_df["Volume"].to_numpy()[weak_caution],
                        marker="x",
                        s=16,
                        color="#f1c40f",
                        label="Weak Caution",
                    )
            if "Volume_Weak_Warning" in price_df.columns:
                weak_warning = price_df["Volume_Weak_Warning"].fillna(False).to_numpy()
                if weak_warning.any():
                    self.volume_ax.scatter(
                        x_indices[weak_warning],
                        price_df["Volume"].to_numpy()[weak_warning],
                        marker="x",
                        s=20,
                        color="#e67e22",
                        label="Weak Warning",
                    )
            if "Volume_Weak_Risk" in price_df.columns:
                weak_risk = price_df["Volume_Weak_Risk"].fillna(False).to_numpy()
                if weak_risk.any():
                    self.volume_ax.scatter(
                        x_indices[weak_risk],
                        price_df["Volume"].to_numpy()[weak_risk],
                        marker="x",
                        s=24,
                        color="#c0392b",
                        label="Weak Risk",
                    )
            self.volume_ax.legend(loc="upper left", fontsize=8, frameon=False)

        # 1.5 Fibonacci Retracement (lookback high/low)
        swing_high = price_df["High"].max()
        swing_low = price_df["Low"].min()
        if swing_high > swing_low:
            fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
            fib_colors = ["#7a7a7a", "#6f6f6f", "#646464", "#595959", "#4e4e4e"]
            for lvl, color in zip(fib_levels, fib_colors):
                price = swing_high - (swing_high - swing_low) * lvl
                self.price_ax.axhline(price, color=color, lw=1.0, ls=":", alpha=0.7)
                self.price_ax.text(
                    1.003,
                    price,
                    f"{lvl:.3f}",
                    transform=self.price_ax.get_yaxis_transform(),
                    color=color,
                    fontsize=7,
                    ha="left",
                    va="center",
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

        # 4. Manually Plot RSI if axis exists
        if self.rsi_ax:
            self.rsi_ax.plot(x_indices, price_df["RSI"], color="black", lw=1.0, label="RSI")
            self.rsi_ax.axhline(70, color="red", lw=0.8, ls="--", alpha=0.5)
            self.rsi_ax.axhline(30, color="green", lw=0.8, ls="--", alpha=0.5)
            self.rsi_ax.set_ylim(0, 100)
            self.rsi_ax.legend(loc="upper left", fontsize=8, frameon=False)
            self.rsi_ax.set_ylabel("RSI")

        # 4. Final Formatting for all visible axes
        all_axes = [ax for ax in [self.price_ax, self.volume_ax, self.macd_ax, self.adx_ax, self.rsi_ax] if ax is not None]
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
        legend_handles = [
            Line2D([0], [0], color="orange", lw=1.0),
            Line2D([0], [0], color="blue", lw=1.0),
        ]
        legend_labels = ["EMA20", "EMA50"]
        if show_supertrend:
            legend_handles.extend([
                Line2D([0], [0], color="green", lw=2.0),
                Line2D([0], [0], color="red", lw=2.0),
            ])
            legend_labels.extend(["ST Bull", "ST Bear"])
        legend_handles.extend([
            Line2D([0], [0], marker='*', color='#00FF00', linestyle='None', markersize=10),
            Line2D([0], [0], marker='*', color='#FF0000', linestyle='None', markersize=10),
        ])
        legend_labels.extend(["Bull Pattern", "Bear Pattern"])
        self.price_ax.legend(legend_handles, legend_labels, loc="upper left", fontsize=9, frameon=True)

        # Last close label near the latest candle
        last_idx = len(price_df) - 1
        last_close = price_df["Close"].iloc[-1]
        self.price_ax.text(
            last_idx + 0.2,
            last_close,
            f"{last_close:,.2f}",
            color="#111111",
            fontsize=8,
            va="center",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#cccccc", alpha=0.8),
        )

        # Hover line/text prepared after autoscale settles
        self._hover_ylim = self.price_ax.get_ylim()
        self._hover_line = self.price_ax.axhline(
            0,
            color="#444444",
            lw=0.8,
            ls=":",
            alpha=0.6,
            visible=False,
        )
        self._hover_text = self.price_ax.text(
            1.003,
            0,
            "",
            transform=self.price_ax.get_yaxis_transform(),
            color="#222222",
            fontsize=8,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#cccccc", alpha=0.8),
            visible=False,
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

    def _on_mouse_move(self, event) -> None:
        if self.current_data is None or event.inaxes != self.price_ax:
            self.status_label.setText(self._base_status_text)
            if self._hover_line:
                self._hover_line.set_visible(False)
            if self._hover_text:
                self._hover_text.set_visible(False)
            return
        if event.xdata is None or event.ydata is None:
            self.status_label.setText(self._base_status_text)
            if self._hover_line:
                self._hover_line.set_visible(False)
            if self._hover_text:
                self._hover_text.set_visible(False)
            return

        y = float(event.ydata)
        self.status_label.setText(f"Price: {y:,.2f}")
        if self._hover_line:
            self._hover_line.set_ydata([y, y])
            self._hover_line.set_visible(True)
        if self._hover_text:
            self._hover_text.set_position((1.003, y))
            self._hover_text.set_text(f"{y:,.2f}")
            self._hover_text.set_visible(True)
        if self._hover_ylim:
            self.price_ax.set_ylim(self._hover_ylim)
        self.canvas.draw_idle()

    def _get_symbol_text(self) -> str:
        return self.symbol_input.currentText().strip().upper()

    def _refresh_symbol_combo(self) -> None:
        current = self.last_symbol if self.last_symbol else "TQQQ"
        self.symbol_input.blockSignals(True)
        self.symbol_input.clear()
        for sym in self.recent_symbols:
            self.symbol_input.addItem(sym)
        if current and current not in self.recent_symbols:
            self.symbol_input.addItem(current)
        self.symbol_input.setCurrentText(current)
        self.symbol_input.blockSignals(False)

    def _remember_symbol(self, symbol: str) -> None:
        if symbol in self.recent_symbols:
            self.recent_symbols.remove(symbol)
        self.recent_symbols.insert(0, symbol)
        self.recent_symbols = self.recent_symbols[:MAX_RECENT_SYMBOLS]
        self.last_symbol = symbol
        self._refresh_symbol_combo()
        self._save_ui_state()

    def _load_ui_state(self) -> None:
        try:
            if UI_STATE_FILE.exists():
                with UI_STATE_FILE.open("r", encoding="utf-8") as f:
                    state = json.load(f)
                recents = state.get("recent_symbols", [])
                self.recent_symbols = [str(s).upper() for s in recents if str(s).strip()]
                self.recent_symbols = self.recent_symbols[:MAX_RECENT_SYMBOLS]
                self.last_symbol = str(state.get("last_symbol", self.last_symbol)).upper()
        except Exception:
            self.recent_symbols = []
            self.last_symbol = "TQQQ"

    def _save_ui_state(self) -> None:
        state = {
            "recent_symbols": self.recent_symbols[:MAX_RECENT_SYMBOLS],
            "last_symbol": self.last_symbol,
        }
        try:
            with UI_STATE_FILE.open("w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=True, indent=2)
        except Exception:
            pass


def main() -> None:
    app = QApplication(sys.argv)
    window = StockTriggerApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
