import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Synthetic seasonal revenue data generation (monthly)
rng = pd.date_range(start='2023-01-01', periods=36, freq='M')
np.random.seed(42)
base = 10_000_000
trend = np.linspace(0, 2_000_000, len(rng))
seasonality = 1_000_000 * np.sin(2 * np.pi * (rng.month - 1) / 12)
noise = np.random.normal(loc=0, scale=300_000, size=len(rng))
revenue = base + trend + seasonality + noise

df = pd.DataFrame({
    'date': rng,
    'year': rng.year,
    'month_num': rng.month,
    'month': rng.month_name().str[:3],
    'revenue': revenue
})

# Styling
sns.set_style('whitegrid')
sns.set_context('talk')

plt.figure(figsize=(8, 8))  # 8in x 8in -> with dpi=64 -> 512x512 px
# Plot monthly revenue time series
sns.lineplot(data=df, x='date', y='revenue', marker='o', linewidth=2.25, label='Monthly Revenue', palette='tab10')

# Overlay a smoothed seasonal average (by month)
seasonal = df.groupby('month_num').revenue.mean().reset_index()
seasonal['month'] = seasonal['month_num'].apply(lambda m: pd.to_datetime(str(m), format='%m').strftime('%b'))
# Map month positions to the middle of each year for visual alignment
month_positions = pd.to_datetime('2023-' + seasonal['month_num'].astype(str) + '-15')
sns.lineplot(x=month_positions, y=seasonal.revenue, data=seasonal, linestyle='--', linewidth=2, label='Avg by Month')

plt.title('Monthly Revenue (Synthetic) — Seasonal Pattern with Trend', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Revenue (USD)')
plt.legend()
plt.tight_layout()
plt.savefig('chart.png', dpi=64, bbox_inches='tight')
