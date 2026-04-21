"""
╔══════════════════════════════════════════════════════════════╗
║       Trading Statistics Analyzer — NSE Edition            ║
║       Author : Shivam Tyagi  |  github.com/QuantShivam     ║
║       Module : 4 — Statistics for Trading                  ║
╚══════════════════════════════════════════════════════════════╝
 
Computes institutional-grade trading statistics on live NSE data.
Every metric is the same one used by professional quant funds to
evaluate whether a strategy is worth trading.
 
Metrics Computed
----------------
Topic 1  — Sharpe Ratio
Topic 2  — Sortino Ratio
Topic 3  — Max Drawdown
Topic 4  — Risk:Reward Ratio
Topic 5  — Kelly Criterion & Half Kelly
Topic 6  — Win Rate & Profit Factor
Topic 7  — Hypothesis Testing (p-value)
Topic 8  — Confidence Intervals
Topic 9  — Regression (stock relationship)
Topic 10 — Distribution Analysis (skew, kurtosis)
Topic 11 — Monte Carlo Simulation (500 paths)
 
Usage
-----
    python analyzer.py
    python analyzer.py --ticker TCS.NS --period 1y
    python analyzer.py --ticker INFY.NS --period 6mo --capital 100000
"""
 
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
import os 
 
 
# ──────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────
 
DEFAULT_TICKER   = "RELIANCE.NS"
DEFAULT_PERIOD   = "1y"
DEFAULT_CAPITAL  = 50_000
RISK_FREE_RATE   = 0.026        # India RFR — 6.5% FD ÷ 252 trading days
N_SIMULATIONS    = 500          # Monte Carlo paths
HIGH_VOL_THRESH  = 1.5          # daily move % flagged as high volatility
 
 
# ──────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────
 
def section(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")
 
 
def verdict(value: float, threshold: float, labels: tuple) -> str:
    return labels[0] if value >= threshold else labels[1]
 
 
# ──────────────────────────────────────────────────────────────
#  DATA DOWNLOAD & PREPARATION
# ──────────────────────────────────────────────────────────────
 
def load_returns(ticker: str, period: str) -> pd.Series:
    """
    Download closing prices and compute daily percentage returns.
    Applies the Module 3 column-flattening fix for yfinance.
    """
    raw = yf.download(ticker, period=period, progress=False)
    raw.columns = [col[0] for col in raw.columns]
    prices  = raw["Close"].ffill().dropna()
    returns = prices.pct_change().dropna() * 100   # as percentages
    print(f"  ✓ {len(returns)} trading days loaded for {ticker}")
    return returns, prices
 
 
# ──────────────────────────────────────────────────────────────
#  TOPIC 1 — Sharpe Ratio
# ──────────────────────────────────────────────────────────────
 
def sharpe_ratio(returns: pd.Series) -> float:
    """
    Return per unit of total risk.
    Formula: (Mean - RFR) / Std × √252
    """
    mean = np.mean(returns)
    std  = np.std(returns)
    return (mean - RISK_FREE_RATE) / std * np.sqrt(252)
 
 
# ──────────────────────────────────────────────────────────────
#  TOPIC 2 — Sortino Ratio
# ──────────────────────────────────────────────────────────────
 
def sortino_ratio(returns: pd.Series) -> float:
    """
    Return per unit of DOWNSIDE risk only.
    Upside volatility (big winning days) is not penalised.
    Formula: (Mean - RFR) / Downside Std × √252
    """
    mean         = np.mean(returns)
    downside_std = returns[returns < 0].std()
    return (mean - RISK_FREE_RATE) / downside_std * np.sqrt(252)
 
 
# ──────────────────────────────────────────────────────────────
#  TOPIC 3 — Max Drawdown
# ──────────────────────────────────────────────────────────────
 
def max_drawdown(returns: pd.Series, capital: float) -> dict:
    """
    Largest peak-to-trough decline in the equity curve.
    Answers: "What was the worst pain at any single moment?"
    """
    equity   = capital * (1 + returns / 100).cumprod()
    peak     = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100
    max_dd   = drawdown.min()
    worst_dt = drawdown.idxmin()
    return {
        "max_drawdown"     : round(max_dd, 2),
        "worst_date"       : str(worst_dt.date()),
        "final_equity"     : round(equity.iloc[-1], 2),
        "equity_curve"     : equity,       # NEW — full array, for charting
        "drawdown_series"  : drawdown,     # NEW — full array, for charting
    }
 
 
# ──────────────────────────────────────────────────────────────
#  TOPIC 4 — Risk:Reward Ratio
# ──────────────────────────────────────────────────────────────
 
def risk_reward(returns: pd.Series) -> dict:
    """
    Ratio of average winning day to average losing day.
    Above 1.0 means wins are on average bigger than losses.
    """
    wins   = returns[returns > 0]
    losses = returns[returns < 0]
    avg_win   = wins.mean()   if len(wins)   > 0 else 0
    avg_loss  = abs(losses.mean()) if len(losses) > 0 else 0
    rr        = avg_win / avg_loss if avg_loss > 0 else 0
    return {
        "avg_win"  : round(avg_win, 3),
        "avg_loss" : round(avg_loss, 3),
        "rr_ratio" : round(rr, 3),
    }
 
 
# ──────────────────────────────────────────────────────────────
#  TOPIC 5 — Kelly Criterion
# ──────────────────────────────────────────────────────────────
 
def kelly_criterion(returns: pd.Series, capital: float) -> dict:
    """
    Optimal position size per trade.
    Always use Half Kelly in real trading — protects against
    losing streaks and inaccurate win rate estimates.
    Formula: Kelly = WinRate - (LossRate / RR)
    """
    rr       = risk_reward(returns)["rr_ratio"]
    win_rate = (returns > 0).mean()
    loss_rate = 1 - win_rate
    kelly     = win_rate - (loss_rate / rr) if rr > 0 else 0
    half_kelly = kelly / 2
    return {
        "win_rate"    : round(win_rate, 3),
        "kelly_pct"   : round(kelly * 100, 2),
        "half_kelly"  : round(half_kelly * 100, 2),
        "risk_amount" : round(capital * half_kelly, 2),
    }
 
 
# ──────────────────────────────────────────────────────────────
#  TOPIC 6 — Win Rate Analysis
# ──────────────────────────────────────────────────────────────
 
def win_rate_analysis(returns: pd.Series) -> dict:
    """
    Win rate, profit factor, and Expected Value.
    Profit Factor = Total gains / Total losses — above 1.0 is profitable.
    EV = average return per day adjusted for win/loss rates.
    """
    wins     = returns[returns > 0]
    losses   = returns[returns < 0]
    total    = len(returns)
    win_rate = len(wins) / total
    pf       = wins.sum() / abs(losses.sum()) if len(losses) > 0 else 0
    ev       = (win_rate * wins.mean()) + ((1 - win_rate) * losses.mean())
    return {
        "win_rate"      : round(win_rate * 100, 1),
        "profit_factor" : round(pf, 3),
        "ev_per_day"    : round(ev, 4),
        "high_vol_days" : int((returns.abs() > HIGH_VOL_THRESH).sum()),
    }
 
 
# ──────────────────────────────────────────────────────────────
#  TOPIC 7 — Hypothesis Testing
# ──────────────────────────────────────────────────────────────
 
def hypothesis_test(returns: pd.Series) -> dict:
    """
    Is the mean return statistically different from zero?
    p-value < 0.05 → edge is unlikely to be random luck.
    """
    t_stat, p_value = stats.ttest_1samp(returns, popmean=0)
    return {
        "t_statistic" : round(t_stat, 3),
        "p_value"     : round(p_value, 4),
        "significant" : p_value < 0.05,
    }
 
 
# ──────────────────────────────────────────────────────────────
#  TOPIC 8 — Confidence Intervals
# ──────────────────────────────────────────────────────────────
 
def confidence_interval(returns: pd.Series) -> dict:
    """
    95% range where the true mean return likely falls.
    Narrow CI = more confident the edge is real.
    """
    n    = len(returns)
    mean = np.mean(returns)
    std  = np.std(returns, ddof=1)
    ci   = stats.t.interval(0.95, df=n-1, loc=mean, scale=std/np.sqrt(n))
    return {
        "mean"     : round(mean, 4),
        "ci_lower" : round(ci[0], 4),
        "ci_upper" : round(ci[1], 4),
        "ci_width" : round(ci[1] - ci[0], 4),
    }
 
 
# ──────────────────────────────────────────────────────────────
#  TOPIC 10 — Distribution Analysis
# ──────────────────────────────────────────────────────────────
 
def distribution_analysis(returns: pd.Series) -> dict:
    """
    Are the returns normally distributed, or do they have
    fat tails and skew? Real markets typically show:
    - Slight negative skew (more crash risk than rally risk)
    - Excess kurtosis (fat tails — extreme events more common)
    """
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)   # excess kurtosis (normal = 0)
    std  = np.std(returns)
    mean = np.mean(returns)
    actual_extreme = (returns.abs() > 2 * std).mean() * 100
    return {
        "skewness"        : round(skew, 3),
        "excess_kurtosis" : round(kurt, 3),
        "extreme_days_%"  : round(actual_extreme, 1),
        "theory_extreme"  : 5.0,
    }
 
 
# ──────────────────────────────────────────────────────────────
#  TOPIC 11 — Monte Carlo Simulation
# ──────────────────────────────────────────────────────────────
 
def monte_carlo(returns: pd.Series,
                capital: float,
                n_sims: int = N_SIMULATIONS) -> dict:
    """
    Run N_SIMULATIONS paths by randomly shuffling historical returns.
    Shows the range of possible outcomes — not just what happened,
    but what COULD happen with a different sequence of the same returns.
    """
    r = returns.values / 100     # convert % to decimal for compounding
    n_days = len(r)
    final_values  = []
    max_drawdowns = []
    all_paths     = []           # NEW — will hold all 500 full equity curves

    np.random.seed(42)
    for _ in range(n_sims):
        sim = np.random.choice(r, size=n_days, replace=True)
        equity   = capital * np.cumprod(1 + sim)
        peak     = np.maximum.accumulate(equity)
        dd       = (equity - peak) / peak * 100
        final_values.append(equity[-1])
        max_drawdowns.append(dd.min())
        all_paths.append(equity)  # NEW — rescue the full curve before the loop forgets it

    fv  = np.array(final_values)
    mdd = np.array(max_drawdowns)
    paths_array = np.array(all_paths)   # NEW — stack all 500 curves into one big 2D grid
    return {
        "best_case"         : round(np.percentile(fv, 95), 2),
        "median"            : round(np.median(fv), 2),
        "worst_case"        : round(np.percentile(fv, 5), 2),
        "median_max_dd"     : round(np.median(mdd), 1),
        "worst_case_max_dd" : round(np.percentile(mdd, 95), 1),
        "pct_profitable"    : round((fv > capital).mean() * 100, 1),
        "n_simulations"     : n_sims,
        "paths"             : paths_array,    # NEW — the rescued 500 paths, ready for charting
    }
 
 
# ──────────────────────────────────────────────────────────────
#  REPORT
# ──────────────────────────────────────────────────────────────
 
def print_report(ticker, period, capital, returns, prices):
 
    sr   = sharpe_ratio(returns)
    so   = sortino_ratio(returns)
    dd   = max_drawdown(returns, capital)
    rr   = risk_reward(returns)
    kc   = kelly_criterion(returns, capital)
    wr   = win_rate_analysis(returns)
    ht   = hypothesis_test(returns)
    ci   = confidence_interval(returns)
    dist = distribution_analysis(returns)
    mc   = monte_carlo(returns, capital)
 
    section(f"TRADING STATISTICS ANALYZER  |  NSE India")
    print(f"  Ticker   : {ticker.replace('.NS','')}")
    print(f"  Period   : {period}  ({len(returns)} trading days)")
    print(f"  Capital  : ₹{capital:,}")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
 
    section("1 & 2 │ SHARPE & SORTINO RATIOS")
    print(f"  Sharpe Ratio  : {sr:.3f}  {'✓ Good' if sr > 1 else '○ Weak' if sr > 0 else '✗ Negative'}")
    print(f"  Sortino Ratio : {so:.3f}  (penalises downside only)")
    print(f"  Sortino > Sharpe: {so > sr}  ({'Expected — more up days' if so > sr else 'Check distribution'})")
    print(f"\n  Interpretation:")
    if sr > 2:
        print("  Excellent strategy — strong return per unit of risk")
    elif sr > 1:
        print("  Good strategy — solid risk-adjusted performance")
    elif sr > 0:
        print("  Weak edge — barely above risk-free rate")
    else:
        print("  Underperforming risk-free rate — strategy needs improvement")
 
    section("3 │ MAX DRAWDOWN")
    print(f"  Max Drawdown  : {dd['max_drawdown']:.2f}%")
    print(f"  Worst Date    : {dd['worst_date']}")
    print(f"  Final Equity  : ₹{dd['final_equity']:,.2f}  (started ₹{capital:,})")
    print(f"  Net Change    : ₹{dd['final_equity'] - capital:+,.2f}")
 
    section("4 & 5 │ RISK:REWARD & KELLY CRITERION")
    print(f"  Avg Win Day   : +{rr['avg_win']:.3f}%")
    print(f"  Avg Loss Day  : -{rr['avg_loss']:.3f}%")
    print(f"  Risk:Reward   : 1:{rr['rr_ratio']:.2f}  ({'✓ Wins > Losses' if rr['rr_ratio'] > 1 else '✗ Losses > Wins'})")
    print(f"\n  Win Rate      : {kc['win_rate']*100:.1f}%")
    print(f"  Kelly %       : {kc['kelly_pct']:.2f}%")
    print(f"  Half Kelly %  : {kc['half_kelly']:.2f}%")
    print(f"  Suggested Risk: ₹{kc['risk_amount']:,.0f} per trade on ₹{capital:,} capital")
 
    section("6 │ WIN RATE ANALYSIS")
    print(f"  Win Rate      : {wr['win_rate']:.1f}%  ({len(returns)} trading days)")
    print(f"  Profit Factor : {wr['profit_factor']:.3f}  ({'✓ Profitable' if wr['profit_factor'] > 1 else '✗ Losing'})")
    print(f"  EV per Day    : {wr['ev_per_day']:+.4f}%")
    print(f"  High Vol Days : {wr['high_vol_days']} days (>±{HIGH_VOL_THRESH}%)")
 
    section("7 & 8 │ HYPOTHESIS TESTING & CONFIDENCE INTERVALS")
    print(f"  Mean Return   : {ci['mean']:.4f}% per day")
    print(f"  T-Statistic   : {ht['t_statistic']:.3f}")
    print(f"  P-Value       : {ht['p_value']:.4f}  ({'✓ Significant edge (p<0.05)' if ht['significant'] else '○ Not significant — need more data'})")
    print(f"  95% CI        : {ci['ci_lower']:.4f}% to {ci['ci_upper']:.4f}%")
    print(f"  CI Width      : {ci['ci_width']:.4f}%  ({'Narrow — high confidence' if ci['ci_width'] < 0.1 else 'Wide — need more data'})")
 
    section("10 │ DISTRIBUTION ANALYSIS")
    print(f"  Skewness      : {dist['skewness']:.3f}  ", end="")
    if dist['skewness'] < -0.5:
        print("(Negative — more crash risk than rally risk)")
    elif dist['skewness'] > 0.5:
        print("(Positive — more rally potential than crash risk)")
    else:
        print("(Roughly symmetric)")
    print(f"  Excess Kurtosis: {dist['excess_kurtosis']:.3f}  ", end="")
    if dist['excess_kurtosis'] > 1:
        print("(Fat tails — extreme events more likely than normal)")
    else:
        print("(Near normal distribution)")
    print(f"  Actual extreme days: {dist['extreme_days_%']:.1f}%  (theory says {dist['theory_extreme']}%)")
 
    section(f"11 │ MONTE CARLO ({mc['n_simulations']} SIMULATIONS)")
    print(f"  Starting Capital  : ₹{capital:,}")
    print(f"  Best case  (95th%): ₹{mc['best_case']:>12,.2f}")
    print(f"  Median            : ₹{mc['median']:>12,.2f}")
    print(f"  Worst case  (5th%): ₹{mc['worst_case']:>12,.2f}")
    print(f"\n  Median Max DD     : {mc['median_max_dd']:.1f}%")
    print(f"  Worst Max DD (95%): {mc['worst_case_max_dd']:.1f}%")
    print(f"  Profitable paths  : {mc['pct_profitable']:.1f}%")
 
    section("SUMMARY VERDICT")
    print(f"  Sharpe       : {sr:.2f}  {'✓' if sr > 1 else '○' if sr > 0 else '✗'}")
    print(f"  Max Drawdown : {dd['max_drawdown']:.1f}%  {'✓' if dd['max_drawdown'] > -20 else '✗'}")
    print(f"  Edge (p<0.05): {'✓ Confirmed' if ht['significant'] else '○ Unconfirmed — more data needed'}")
    print(f"  Profitable MC: {mc['pct_profitable']:.0f}% of simulations")
    if sr > 1 and ht['significant'] and mc['pct_profitable'] > 60:
        print("\n  ▶ OVERALL: Strong statistical case for trading this strategy")
    elif sr > 0 and mc['pct_profitable'] > 50:
        print("\n  ▶ OVERALL: Weak edge — trade small, monitor closely")
    else:
        print("\n  ▶ OVERALL: Insufficient edge — do not trade until edge improves")
 

    section("GENERATING CHARTS")

    # Create an 'images' folder next to the script if it doesn't exist.
    # exist_ok=True means "don't complain if the folder is already there."
    os.makedirs("images", exist_ok=True)

    # Apply a clean, professional style to all charts at once.
    # This is like picking the theme for a PowerPoint deck.
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "DejaVu Sans"

    # Pull out the data we rescued in Moves 1 and 2.
    # dd and mc are dictionaries returned earlier in print_report.
    equity_curve    = dd["equity_curve"]
    drawdown_series = dd["drawdown_series"]
    mc_paths        = mc["paths"]

    ticker_clean = ticker.replace(".NS", "")

    # ─── CHART 1: Equity Curve ───
    # Answers: "Did the strategy make money, and how smoothly?"
    plt.figure(figsize=(10, 5))                                  # canvas
    plt.plot(equity_curve.values, color="#2E7D32", linewidth=2)  # draw
    plt.title(f"Equity Curve — {ticker_clean} — "                # label
              f"₹{capital:,.0f} starting capital",
              fontsize=13, fontweight="bold")
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value (₹)")
    plt.tight_layout()
    plt.savefig("images/equity_curve.png", dpi=150)              # save
    plt.close()                                                  # close
    print("  ✓ images/equity_curve.png saved")

    # ─── CHART 2: Drawdown Chart ───
    # Answers: "How painful were the worst stretches?"
    plt.figure(figsize=(10, 5))
    # fill_between shades the area between the drawdown line and zero.
    plt.fill_between(range(len(drawdown_series)),
                     drawdown_series.values, 0,
                     color="#C62828", alpha=0.4)
    plt.plot(drawdown_series.values, color="#C62828", linewidth=1.5)
    plt.title(f"Drawdown — Max: {dd['max_drawdown']:.2f}%",
              fontsize=13, fontweight="bold")
    plt.xlabel("Trading Days")
    plt.ylabel("Drawdown from Peak (%)")
    plt.tight_layout()
    plt.savefig("images/drawdown.png", dpi=150)
    plt.close()
    print("  ✓ images/drawdown.png saved")

    # ─── CHART 3: Returns Distribution Histogram ───
    # Answers: "Are returns normal, or do extreme days happen too often?"
    plt.figure(figsize=(10, 5))
    plt.hist(returns.values, bins=40, color="#1565C0", alpha=0.7,
             edgecolor="white", density=True, label="Actual returns")

    # Overlay a reference normal distribution curve.
    x_range     = np.linspace(returns.min(), returns.max(), 200)
    normal_pdf  = stats.norm.pdf(x_range, returns.mean(), returns.std())
    plt.plot(x_range, normal_pdf, color="#EF6C00", linewidth=2,
             label="Normal distribution (reference)")

    plt.title(f"Daily Returns Distribution — {ticker_clean}",
              fontsize=13, fontweight="bold")
    plt.xlabel("Daily Return (%)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/returns_distribution.png", dpi=150)
    plt.close()
    print("  ✓ images/returns_distribution.png saved")

    # ─── CHART 4: Monte Carlo Fan ───
    # Answers: "What's the range of possible futures for this strategy?"
    plt.figure(figsize=(10, 5))

    # Plot all 500 simulated paths in ONE vectorized call.
    # mc_paths.T means "transpose" — flipping rows and columns so that
    # matplotlib reads each column as one line to draw. Passing the whole
    # array at once is roughly 50× faster than a Python for-loop.
    plt.plot(mc_paths.T, color="grey", alpha=0.05, linewidth=0.5)

    # Overlay the 5th, 50th, and 95th percentile paths in bold colors.
    # axis=0 means "compute percentile across simulations, day by day."
    p5  = np.percentile(mc_paths, 5,  axis=0)
    p50 = np.percentile(mc_paths, 50, axis=0)
    p95 = np.percentile(mc_paths, 95, axis=0)

    plt.plot(p95, color="#2E7D32", linewidth=2, label="95th percentile (best)")
    plt.plot(p50, color="#1565C0", linewidth=2, label="Median outcome")
    plt.plot(p5,  color="#C62828", linewidth=2, label="5th percentile (worst)")

    plt.title(f"Monte Carlo Simulation — {mc['n_simulations']} Paths",
              fontsize=13, fontweight="bold")
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value (₹)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/monte_carlo_fan.png", dpi=150)
    plt.close()
    print("  ✓ images/monte_carlo_fan.png saved")
    section("END OF REPORT")
    print(f"  Author: Shivam Tyagi  |  github.com/QuantShivam\n")
 
 
# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────
 
def parse_args():
    p = argparse.ArgumentParser(description="Trading Statistics Analyzer")
    p.add_argument("--ticker",  default=DEFAULT_TICKER)
    p.add_argument("--period",  default=DEFAULT_PERIOD)
    p.add_argument("--capital", default=DEFAULT_CAPITAL, type=float)
    return p.parse_args()
 
 
def main():
    args    = parse_args()
    returns, prices = load_returns(args.ticker, args.period)
    print_report(args.ticker, args.period, args.capital, returns, prices)
 
 
if __name__ == "__main__":
    main()