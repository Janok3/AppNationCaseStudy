import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the data
df = pd.read_csv('Question 1 Analysis.csv')

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Derived Metrics Analysis
# ARPU by platform
df['ARPU'] = df['revenue'] / df['new_installs']
arpu_by_platform = df.groupby('platform')['ARPU'].mean()

# ROI and CAC by channel
# ROI and CAC by channel with zero handling
df['ROI'] = df.apply(lambda row: (row['revenue'] - row['ad_cost']) / row['ad_cost'] if row['ad_cost'] != 0 else 0, axis=1)
df['CAC'] = df.apply(lambda row: row['ad_cost'] / row['new_installs'] if row['new_installs'] != 0 else 0, axis=1)
roi_by_channel = df.groupby('channel')['ROI'].mean()
cac_by_channel = df.groupby('channel')['CAC'].mean()

# Print derived metrics
print("ARPU by Platform:")
print(arpu_by_platform)
print("\nROI by Channel:")
print(roi_by_channel)
print("\nCAC by Channel:")
print(cac_by_channel)

# Time-Series Trend Analysis
df['month'] = df['date'].dt.to_period('M')
monthly_installs = df.groupby(['month', 'platform'])['new_installs'].sum().unstack()

# Plot monthly installs
monthly_installs.plot(kind='line')
plt.title('Monthly Installs by Platform')
plt.xlabel('Month')
plt.ylabel('Installs')
plt.grid(True)
plt.legend(title='Platform')
plt.show()

# ROI Visualization by Channel and Month
monthly_roi = df.groupby(['month', 'channel'])['ROI'].mean().unstack()

# Plot grouped bar chart
monthly_roi.plot(kind='bar')
plt.title('Monthly ROI by Channel')
plt.xlabel('Month')
plt.ylabel('ROI')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend(title='Channel')
plt.show()

# Interpretation and Recommendations
print("\nAnalysis Insights:")
print(f"Platform with higher ARPU: {arpu_by_platform.idxmax()}")
print(f"Channel with highest ROI: {roi_by_channel.idxmax()}")
print(f"Channel with lowest CAC: {cac_by_channel.idxmin()}")

# Additional insights
print("\nRecommendations:")
print("1. Allocate more budget to channels with highest ROI and lowest CAC")
print("2. Investigate platform-specific user behavior to optimize ARPU")
print("3. Monitor monthly trends to adjust marketing strategies")
print("4. Consider seasonal factors in user acquisition planning")
