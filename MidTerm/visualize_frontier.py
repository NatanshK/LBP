import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ga_results = pd.read_csv("ga_pareto_front.csv")
ga_results = ga_results.sort_values(by="Shortfall") 


returns_df = pd.read_csv("asset_log_returns.csv", index_col=0)
market_data = returns_df.values
mean_returns = np.mean(market_data, axis=0)

n_assets = market_data.shape[1]
n_days = market_data.shape[0]
k = int(0.05 * n_days) 


n_portfolios = 10000
unconstrained_returns = []
unconstrained_risks = []

print("Simulating 10,000 unconstrained portfolios for the baseline...")
for _ in range(n_portfolios):
    weights = np.random.dirichlet(np.ones(n_assets))
    

    port_ret = np.sum(weights * mean_returns)
    unconstrained_returns.append(port_ret)
    

    daily_returns = market_data.dot(weights)
    sorted_returns = np.sort(daily_returns)
    tail_loss = -np.mean(sorted_returns[:k])
    unconstrained_risks.append(tail_loss)


plt.figure(figsize=(12, 8))


plt.scatter(unconstrained_risks, unconstrained_returns, color='lightgray', alpha=0.5, 
            label='Unconstrained Market (All 50 Stocks)')


plt.plot(ga_results['Shortfall'], ga_results['Return'], color='red', marker='o', 
         linestyle='-', linewidth=2, markersize=5, 
         label='AI Pareto Front (Constrained to 10 Stocks)')


plt.title("C++ Genetic Algorithm vs. Unconstrained Market Baseline", fontsize=16)
plt.xlabel("Risk: Expected Shortfall (CVaR)", fontsize=12)
plt.ylabel("Reward: Expected Daily Return", fontsize=12)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)


plt.annotate('Better Portfolios\n(High Return, Low Risk)', xy=(0.02, 0.0009), xytext=(0.025, 0.0007),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=10, ha='center')

plt.show()