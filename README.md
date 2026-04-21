# Trading Statistics Analyzer

**Know if your trading strategy has a real edge — or if you've just been getting lucky.**

A statistical analysis engine for Indian equity strategies. Runs ten professional quant metrics on your trading data and gives you a single verdict: is this strategy worth real capital, or not?

![Equity Curve Sample](images/equity_curve.png)

---

## Who this is for

**Retail options and equity traders in India** who have at least six months of trade history and want to know — with real statistics, not gut feel — whether their edge is genuine or random.

**Trading educators** who teach strategies and want to validate them with rigorous numbers before publishing or selling courses.

**Small PMS analysts and prop desk researchers** who need fast Monte Carlo risk analysis and Sharpe/Sortino reporting without building the infrastructure from scratch.

If you've ever looked at your trading P&L and wondered *"is this skill or luck?"* — this tool answers that question.

---

## What it does

Ten institutional-grade statistical tests on any NSE equity or strategy return series:

- **Sharpe & Sortino Ratios** — risk-adjusted return, penalising only downside
- **Max Drawdown** — worst peak-to-trough loss in full equity history
- **Risk : Reward & Win Rate** — trade quality at a glance
- **Kelly Criterion (Half Kelly default)** — optimal position sizing
- **Hypothesis Testing (p-value)** — is your edge statistically real, or noise?
- **95% Confidence Intervals** — how confident we are about the true mean return
- **Distribution Analysis (skew & kurtosis)** — fat tails and crash risk detection
- **Monte Carlo Simulation (500 paths)** — worst-case scenarios and outcome ranges

Every metric comes with a plain-English verdict. No black boxes. All formulas in the code.