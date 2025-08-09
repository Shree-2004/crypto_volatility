import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("dataset.csv")

# Remove infinite & missing values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Parse date column
df['date'] = pd.to_datetime(df['date'])

# Sort by crypto and date
df = df.sort_values(by=['crypto_name', 'date'])

# Feature engineering
df['volatility'] = (df['high'] - df['low']) / df['open']
df['liquidity_ratio'] = df['volume'] / df['marketCap']
df['rolling_volatility'] = df.groupby('crypto_name')['volatility'].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)

# Save output
df.to_csv("processed_dataset.csv", index=False)
print("✅ Saved: processed_dataset.csv")
