import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# Generate synthetic seasonal revenue data
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# Professional Seaborn styling
# ---------------------------------------------------------
sns.set_style("whitegrid")
sns.set_context("talk")

# ---------------------------------------------------------
# Create figure EXACTLY 512×512 pixels
# 8 inches × 64 DPI = 512 pixels
# ---------------------------------------------------------
plt.figure(figsize=(8, 8))

# Main lineplot
sns.lineplot(
    data=df,
    x='date',
    y='revenue',
    marker='o',
    linewidth=2.5,
    label='Monthly Revenue'
)

# Seasonal average overlay
seasonal = df.groupby('month_num').revenue.mean().reset_index()
seasonal['month'] = seasonal['month_num'].apply(lambda m: pd.to_datetime(str(m), format='%m').strftime('%b'))
seasonal_x = pd.to_datetime('2023-' + seasonal['month_num'].astype(str) + '-15')

sns.lineplot(
    x=seasonal_x,
    y=seasonal['revenue'],
    linestyle='--',
    linewidth=2,
    label='Avg Seasonal Revenue'
)

plt.title("Monthly Revenue — Seasonal Trend (Synthetic)", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Revenue (USD)")
plt.legend()

# ---------------------------------------------------------
# SAVE IMAGE — EXACT SIZE GUARANTEED
# ---------------------------------------------------------
plt.savefig("chart.png", dpi=64)   # IMPORTANT: NO bbox_inches='tight'
plt.close()
