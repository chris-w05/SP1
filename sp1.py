import os
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import warnings
warnings.filterwarnings('ignore')

try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

SP500_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "BRK-B", "LLY", "AVGO",
    "TSLA", "JPM", "V", "UNH", "XOM", "MA", "PG", "JNJ", "HD", "MRK", "COST",
    "ABBV", "NFLX", "AMD", "CRM", "TMO", "LIN", "ADBE", "MCD", "CSCO", "ACN"
]

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sp1_cache")
SHARES_MAX_AGE_DAYS = 30   


# ── Cache I/O ─────────────────────────────────────────────────────────────────

def _price_path(ticker):
    return os.path.join(CACHE_DIR, "prices", f"{ticker}.csv")

def _shares_path(ticker):
    return os.path.join(CACHE_DIR, "shares", f"{ticker}.csv")

def _read_series(path):
    if not os.path.exists(path):
        return None
    try:
        s = pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns")
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s.sort_index().astype(float)
    except Exception:
        return None

def _write_series(series, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    series.to_csv(path, header=True)


# ── Price data (cached, gap-filling) ─────────────────────────────────────────

def _download_prices(ticker, start, end):
    try:
        dl_end = pd.Timestamp(end) + pd.DateOffset(months=2)
        df = yf.download(ticker,
                         start=pd.Timestamp(start).strftime("%Y-%m-%d"),
                         end=dl_end.strftime("%Y-%m-%d"),
                         interval="1mo", progress=False,
                         auto_adjust=True, threads=False)
        if df.empty:
            return None
        s = df["Close"].squeeze()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s.sort_index().astype(float)
    except Exception:
        return None


def get_price_series(ticker, start_date, end_date):
    path   = _price_path(ticker)
    cached = _read_series(path)
    start  = pd.Timestamp(start_date)
    end    = pd.Timestamp(end_date)

    parts = []

    if cached is not None and not cached.empty:
        cmin, cmax = cached.index.min(), cached.index.max()

        if start < cmin:                          
            pre = _download_prices(ticker, start, cmin - pd.DateOffset(days=1))
            if pre is not None and not pre.empty:
                parts.append(pre)

        parts.append(cached)

        if end > cmax:                            
            post = _download_prices(ticker, cmax + pd.DateOffset(days=1), end)
            if post is not None and not post.empty:
                parts.append(post)
    else:
        full = _download_prices(ticker, start, end)
        if full is not None and not full.empty:
            parts.append(full)

    if not parts:
        return None

    combined = pd.concat(parts).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    _write_series(combined, path)

    return combined[(combined.index >= start) & (combined.index <= end)]


# ── Shares data (cached, age-based refresh) ───────────────────────────────────

def _download_shares(ticker, start_date):
    try:
        t = yf.Ticker(ticker)
        try:
            raw = t.get_shares_full(start=start_date)
            if raw is not None and len(raw) > 0:
                idx = pd.to_datetime(raw.index)
                if idx.tzinfo is not None:
                    idx = idx.tz_localize(None)
                return pd.Series(raw.values.astype(float), index=idx).sort_index()
        except Exception:
            pass

        bs = t.quarterly_balance_sheet
        for row_name in ["Ordinary Shares Number", "Share Issued", "Common Stock"]:
            if row_name in bs.index:
                row = bs.loc[row_name].dropna()
                if not row.empty:
                    idx = pd.to_datetime(row.index)
                    if idx.tzinfo is not None:
                        idx = idx.tz_localize(None)
                    return pd.Series(row.values.astype(float), index=idx).sort_index()
    except Exception:
        pass
    return None


def get_shares_series(ticker, start_date):
    path = _shares_path(ticker)

    if os.path.exists(path):
        age_days = (datetime.now().timestamp() - os.path.getmtime(path)) / 86400
        if age_days < SHARES_MAX_AGE_DAYS:
            cached = _read_series(path)
            if cached is not None and not cached.empty:
                return cached

    series = _download_shares(ticker, start_date)
    if series is not None and not series.empty:
        _write_series(series, path)
    return series


def pit_shares(history, ticker, as_of):
    if ticker not in history:
        return None
    s = history[ticker]
    prior = s[s.index <= as_of]
    return float(prior.iloc[-1]) if not prior.empty else float(s.iloc[0])


# ── Aggregate loaders ─────────────────────────────────────────────────────────

def load_prices(start_date, end_date, status_cb=None):
    closes = {}
    n = len(SP500_TICKERS)
    for i, ticker in enumerate(SP500_TICKERS):
        if status_cb:
            status_cb(f"Loading prices… ({i + 1}/{n})")
        s = get_price_series(ticker, start_date, end_date)
        if s is not None and not s.empty:
            closes[ticker] = s

    if not closes:
        raise ValueError("Failed to load any price data.")

    prices = pd.DataFrame(closes).sort_index().ffill().bfill()
    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
    return prices[(prices.index >= start_ts) & (prices.index <= end_ts)]


def load_shares(start_date, status_cb=None):
    history = {}
    n = len(SP500_TICKERS)
    for i, ticker in enumerate(SP500_TICKERS):
        if status_cb:
            status_cb(f"Loading shares… ({i + 1}/{n})")
        s = get_shares_series(ticker, start_date)
        if s is not None and not s.empty:
            history[ticker] = s
    return history


# ── Simulation ────────────────────────────────────────────────────────────────

def run_simulation(start_str, end_str, seed, monthly_contrib, status_cb=None):
    start_date = pd.to_datetime(start_str)
    end_date   = pd.to_datetime(end_str)

    if start_date >= end_date:
        raise ValueError("Start date must be before end date.")

    prices         = load_prices(start_date, end_date, status_cb=status_cb)
    shares_history = load_shares(start_date, status_cb=status_cb)

    if status_cb:
        status_cb("Loading SPY benchmark…")
    spy_raw     = get_price_series("SPY", start_date, end_date)
    dates       = prices.index
    spy         = spy_raw.reindex(dates, method="ffill")
    spy_returns = spy.pct_change()

    value_sp500  = float(seed)
    value_sp1    = float(seed)
    sp500_values = []
    sp1_values   = []
    top_tickers  = [] 

    if status_cb:
        status_cb("Running simulation…")

    for i in range(len(dates)):
        value_sp500 += monthly_contrib
        value_sp1   += monthly_contrib

        if i > 0:
            r = spy_returns.iloc[i]
            if pd.notna(r):
                value_sp500 *= (1 + r)
        sp500_values.append(value_sp500)

        if i == 0:
            sp1_values.append(value_sp1)
            top_tickers.append(None)
            continue

        month_start = dates[i - 1]
        prev_prices = prices.iloc[i - 1]

        market_caps = {}
        for t in SP500_TICKERS:
            if t not in prev_prices.index or pd.isna(prev_prices[t]):
                continue
            shares = pit_shares(shares_history, t, month_start)
            if shares and shares > 0:
                market_caps[t] = prev_prices[t] * shares

        if not market_caps:
            sp1_values.append(value_sp1)
            top_tickers.append(top_tickers[-1] if len(top_tickers) > 0 else None)
            continue

        top_ticker  = max(market_caps, key=market_caps.get)
        top_tickers.append(top_ticker)

        p0 = prices.iloc[i - 1][top_ticker]
        p1 = prices.iloc[i][top_ticker]
        if pd.notna(p0) and pd.notna(p1) and p0 > 0:
            value_sp1 *= p1 / p0

        sp1_values.append(value_sp1)

    return dates, np.array(sp500_values), np.array(sp1_values), top_tickers


# ── GUI ───────────────────────────────────────────────────────────────────────

class InvestmentSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("S&P 1 Strategy Simulator")
        self.root.geometry("1100x680")
        self.root.configure(bg="#f0f2f5")
        self._build_ui()

    def _build_ui(self):
        left = tk.Frame(self.root, bg="white", width=230)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0), pady=10)
        left.pack_propagate(False)

        tk.Label(left, text="S&P 1 Simulator", font=("Arial", 13, "bold"),
                 bg="white", fg="#1a1a2e").pack(pady=(16, 2))
        tk.Label(left, text="Largest Market Cap vs. Index", font=("Arial", 9),
                 bg="white", fg="#888").pack(pady=(0, 12))
        tk.Frame(left, bg="#e0e0e0", height=1).pack(fill=tk.X, padx=12)

        default_end = datetime.now().strftime("%Y-%m")
        fields = [
            ("Start Date (YYYY-MM)", "start_date", "2015-01"),
            ("End Date   (YYYY-MM)", "end_date",   default_end),
            ("Initial Seed ($)",     "seed",       "10000"),
            ("Monthly Contribution ($)", "monthly", "1000"),
        ]
        self.entries = {}
        for label_text, key, default in fields:
            tk.Label(left, text=label_text, font=("Arial", 9),
                     bg="white", fg="#555", anchor="w").pack(
                fill=tk.X, padx=14, pady=(10, 2))
            e = tk.Entry(left, font=("Arial", 10), relief="solid", bd=1)
            e.insert(0, default)
            e.pack(fill=tk.X, padx=14, ipady=4)
            self.entries[key] = e

        tk.Button(left, text="▶  Run Simulation",
                  font=("Arial", 10, "bold"),
                  bg="#1976D2", fg="white", activebackground="#1565C0",
                  relief="flat", cursor="hand2", pady=8,
                  command=self.run).pack(fill=tk.X, padx=14, pady=16)

        tk.Frame(left, bg="#e0e0e0", height=1).pack(fill=tk.X, padx=12)
        tk.Label(left, text="RESULTS", font=("Arial", 8, "bold"),
                 bg="white", fg="#aaa").pack(anchor="w", padx=14, pady=(10, 4))

        self.result_labels = {}
        for key, label in [("sp1",  "S&P 1 Final"),
                            ("sp500","S&P 500 Final"),
                            ("top",  "Current Holding"),
                            ("gain", "Difference")]:
            tk.Label(left, text=label, font=("Arial", 8),
                     bg="white", fg="#888").pack(anchor="w", padx=14)
            lbl = tk.Label(left, text="—", font=("Arial", 10, "bold"),
                           bg="white", fg="#333")
            lbl.pack(anchor="w", padx=14, pady=(0, 6))
            self.result_labels[key] = lbl

        right = tk.Frame(self.root, bg="#f0f2f5")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(9, 5.2))
        self.fig.patch.set_facecolor("#f0f2f5")
        
        self.canvas = FigureCanvasTkAgg(self.fig, right)
        
        # FIX: The toolbar must be created AND explicitly packed to the bottom
        self.toolbar = NavigationToolbar2Tk(self.canvas, right)
        self.toolbar.configure(bg="#f0f2f5")
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X) # THIS FORCES IT TO SHOW
        
        # Then we pack the canvas into the remaining space above the toolbar
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _set_status(self, msg):
        self.result_labels["sp1"].config(text=msg, fg="#555")
        self.root.update()

    def run(self):
        try:
            start_str = self.entries["start_date"].get().strip()
            end_str   = self.entries["end_date"].get().strip()
            seed      = float(self.entries["seed"].get())
            monthly   = float(self.entries["monthly"].get())

            self._set_status("Starting…")

            dates, sp500_vals, sp1_vals, top_tickers = run_simulation(
                start_str, end_str, seed, monthly, status_cb=self._set_status)

            final_sp500 = sp500_vals[-1]
            final_sp1   = sp1_vals[-1]
            final_top   = top_tickers[-1]
            diff        = final_sp1 - final_sp500
            diff_pct    = (final_sp1 / final_sp500 - 1) * 100
            sign        = "+" if diff >= 0 else ""
            green, red  = "#27ae60", "#c0392b"

            self.result_labels["sp1"].config(
                text=f"${final_sp1:,.0f}",
                fg=green if final_sp1 >= final_sp500 else red)
            self.result_labels["sp500"].config(text=f"${final_sp500:,.0f}", fg="#333")
            self.result_labels["top"].config(text=final_top or "—", fg="#1976D2")
            self.result_labels["gain"].config(
                text=f"{sign}${diff:,.0f} ({sign}{diff_pct:.1f}%)",
                fg=green if diff >= 0 else red)

            self.ax.clear()
            self.ax.set_facecolor("white")
            
            # --- Draw the standard SP500 line ---
            self.ax.plot(dates, sp500_vals, label="S&P 500 (SPY)",
                         linewidth=2, color="#1976D2")

            # --- Map Unique Tickers to Colors ---
            unique_tickers = list(set([t for t in top_tickers if t is not None]))
            cmap = plt.get_cmap('tab10') 
            ticker_colors = {t: cmap(i % 10) for i, t in enumerate(unique_tickers)}

            self.ax.plot([], [], label="S&P 1 (Largest Market Cap)", linewidth=2.5, color="gray")

            # --- Break SP1 into colored segments ---
            blocks = []
            if len(dates) > 1:
                start_i = 1
                curr_t = top_tickers[1]
                for i in range(2, len(dates)):
                    if top_tickers[i] != curr_t:
                        blocks.append((curr_t, start_i, i))
                        curr_t = top_tickers[i]
                        start_i = i
                blocks.append((curr_t, start_i, len(dates)))

            # Plot each block with its respective color and label
            for block_index, (t, s, e) in enumerate(blocks):
                if t is None: continue
                seg_x = dates[s-1 : e]
                seg_y = sp1_vals[s-1 : e]
                c = ticker_colors[t]
                
                self.ax.plot(seg_x, seg_y, linewidth=2.5, color=c)
                
                mid_idx = len(seg_x) // 2
                x_pos = seg_x[mid_idx]
                y_pos = seg_y[mid_idx]
                
                if block_index % 2 == 0:
                    y_offset = 12
                    v_align = 'bottom'
                else:
                    y_offset = -12
                    v_align = 'top'
                
                self.ax.annotate(t, xy=(x_pos, y_pos), xytext=(0, y_offset),
                                 textcoords="offset points", color=c,
                                 fontsize=8, fontweight='bold', ha='center', va=v_align,
                                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

            # --- Fills ---
            self.ax.fill_between(dates, sp500_vals, sp1_vals,
                                 where=(sp1_vals >= sp500_vals),
                                 alpha=0.07, color=green)
            self.ax.fill_between(dates, sp500_vals, sp1_vals,
                                 where=(sp1_vals < sp500_vals),
                                 alpha=0.07, color=red)

            self.ax.set_title("S&P 1 (Largest Market Cap) vs. S&P 500",
                              fontsize=13, fontweight="bold", pad=10)
            self.ax.set_ylabel("Portfolio Value ($)", fontsize=10)
            self.ax.legend(fontsize=10, framealpha=1.0, edgecolor="#ddd")
            self.ax.grid(True, alpha=0.2, linestyle="--")
            self.ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            self.ax.tick_params(axis='both', labelsize=8)
            
            self.fig.tight_layout()
            self.canvas.draw()
            
            # Tells the toolbar to register the new bounds for the pan/zoom features
            self.toolbar.update() 

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.0)
    except Exception:
        pass
    app = InvestmentSimulator(root)
    root.mainloop()