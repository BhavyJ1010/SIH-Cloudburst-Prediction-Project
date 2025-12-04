import pandas as pd
df = pd.read_csv('/Users/namrata/Namrata/cloudburst_analysis/synthetic_cloudburst_data (1).csv', parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
df.head()
df.info()
df.isna().sum()
df['pressure'] = df['pressure'].interpolate(limit=5)
df[['rainfall_mm','rain_last_5min','rain_last_15min']] = df[['rainfall_mm','rain_last_5min','rain_last_15min']].fillna(0)

df = df.set_index('timestamp')
df['rain_5'] = df['rainfall_mm'].rolling('5min').sum()
df['rain_15'] = df['rainfall_mm'].rolling('15min').sum()
df['pressure_prev_15'] = df['pressure'].shift(15)   
df['pressure_drop_15'] = df['pressure_prev_15'] - df['pressure']
df['humidity_prev_15'] = df['humidity'].shift(15)
df['humidity_change_15'] = df['humidity'] - df['humidity_prev_15']
df = df.dropna().reset_index()
df.to_csv('engineered_features.csv', index=False)

df = df.drop(columns=['radar_dbz'], errors='ignore')
print(df.info())


import matplotlib.pyplot as plt
plt.figure(figsize=(12,3))
plt.plot(df['timestamp'], df['rain_15'], label='rain_15min')
plt.legend(); plt.savefig('rain_15_timeseries.png')


evt = df[df['cloudburst']==1]
plt.figure(figsize=(12,3))
plt.plot(df['timestamp'], df['rain_15'], label='rain_15min')
plt.scatter(evt['timestamp'], evt['rain_15'], color='red', label='cloudburst')
plt.legend(); plt.savefig('rain_cloudburst_overlay.png')

import seaborn as sns
sns.histplot(data=df, x='rain_15', hue='cloudburst', stat='density', common_norm=False)
plt.savefig('rain15_hist_by_label.png')

corr = df[['rain_15','rain_5','pressure_drop_15','humidity_change_15','satellite_precip','forecast_30min','cloudburst']].corr()
sns.heatmap(corr, annot=True, fmt=".2f")
plt.savefig('corr_heatmap.png')

pos = df[df['cloudburst']==1]
neg = df[df['cloudburst']==0]
for col in ['rain_15','pressure_drop_15','humidity_change_15']:
    print(col)
    print('pos p50,p75,p90:', pos[col].quantile([0.5,0.75,0.9]).to_dict())
    print('neg p50,p75,p90:', neg[col].quantile([0.5,0.75,0.9]).to_dict())
