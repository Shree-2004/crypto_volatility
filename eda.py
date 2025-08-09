import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load your dataset (make sure it's the original one)
df = pd.read_csv("dataset.csv")

# Basic cleaning
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Filter only for Bitcoin (or change to another coin if needed)
btc_df = df[df['crypto_name'] == 'Bitcoin'].copy()
btc_df = btc_df.sort_values(by='date')

# Feature engineering
btc_df['volatility'] = (btc_df['high'] - btc_df['low']) / btc_df['open']
btc_df['liquidity_ratio'] = btc_df['volume'] / btc_df['marketCap']
btc_df['rolling_volatility'] = btc_df['volatility'].rolling(7, min_periods=1).mean()

# Create a folder to save the plots
output_dir = "eda_outputs"
os.makedirs(output_dir, exist_ok=True)

# 1. Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
    btc_df[['open', 'high', 'low', 'close', 'volume', 'marketCap', 'volatility', 'liquidity_ratio', 'rolling_volatility']].corr(),
    annot=True, cmap="coolwarm", fmt=".2f"
)
plt.title("Bitcoin Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "btc_correlation_heatmap.png"))
plt.close()

# 2. Volatility Trend
plt.figure(figsize=(10, 4))
plt.plot(btc_df['date'], btc_df['volatility'], label='Daily Volatility', color='red', alpha=0.6)
plt.plot(btc_df['date'], btc_df['rolling_volatility'], label='7-day Rolling Volatility', color='blue')
plt.title("Bitcoin Daily Volatility Trend")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "btc_volatility_trend.png"))
plt.close()

# 3. Volatility Distribution
plt.figure(figsize=(8, 4))
sns.histplot(btc_df['volatility'], bins=50, kde=True, color='purple')
plt.title("Distribution of Daily Volatility - Bitcoin")
plt.xlabel("Volatility")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "btc_volatility_distribution.png"))
plt.close()

print("✅ EDA visualizations saved in the 'eda_outputs' folder.")
