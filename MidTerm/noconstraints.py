import pandas as pd
import numpy as np
import scipy.optimize as sco

df = pd.read_csv("asset_log_returns.csv", index_col=0)

# Annualize the data for the Markowitz solver
mean_returns = df.mean() * 252
cov_matrix = df.cov() * 252
num_assets = len(df.columns)
tickers = df.columns.tolist()

def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights)
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std_dev

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    p_ret, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return - (p_ret - risk_free_rate) / p_std

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0.0, 1.0) for _ in range(num_assets))
initial_guess = num_assets * [1. / num_assets]

print("Running Unconstrained Markowitz Optimization (SLSQP)...")
optimized_result = sco.minimize(
    negative_sharpe_ratio, 
    initial_guess, 
    args=(mean_returns, cov_matrix), 
    method='SLSQP', 
    bounds=bounds, 
    constraints=constraints
)

best_weights = optimized_result.x


opt_return, opt_risk = portfolio_performance(best_weights, mean_returns, cov_matrix)
max_sharpe = opt_return / opt_risk


daily_opt_return = opt_return / 252.0
daily_opt_risk = opt_risk / np.sqrt(252.0)

print("\n  THE UNCONSTRAINED MARKOWITZ RESULTS")
print("  (Maximum Sharpe Ratio Portfolio)\n")
print(f"Expected Annual Return:  {opt_return * 100:.2f}%")
print(f"Annual Volatility:       {opt_risk * 100:.2f}%")
print(f"Sharpe Ratio:            {max_sharpe:.3f}")
print(f"Expected Daily Return:   {daily_opt_return:.6f} ({daily_opt_return * 100:.4f}%)")
print(f"Daily Volatility:        {daily_opt_risk:.6f} ({daily_opt_risk * 100:.4f}%)")


print("Exact Asset Allocation")
for i in range(num_assets):
    if best_weights[i] > 0.01:
        print(f"{tickers[i]}: {best_weights[i] * 100:.2f}%")