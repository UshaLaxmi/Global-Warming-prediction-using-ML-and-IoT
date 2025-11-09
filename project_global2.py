import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.ticker as ticker  #  for clean y-axis tick formatting

# Step 1: Load Excel and clean
df = pd.read_excel('feedsss.xlsx')
df.columns = df.columns.str.strip()  # Strip column name spaces
df = df.dropna(subset=['entry_id', 'area', 'field1', 'field2', 'field3'])

# Step 2: Extract values
entry_ids = df['entry_id'].astype(str)
areas = df['area'].astype(str)
temperature = df['field1']
humidity = df['field2']
aqi = df['field3']

# Step 3: Combine labels like "1 - Charminar"
x_labels = entry_ids + " - " + areas

# Step 4: Calculate global warming %
baseline_temp = 25.0
global_warming_percent = (temperature - baseline_temp) / baseline_temp * 100
average_gw_percent = global_warming_percent.mean()

# Step 5: Linear regression on temperature
x = np.arange(len(df)).reshape(-1, 1)
model = LinearRegression()
model.fit(x, temperature)
temperature_pred = model.predict(x)

# Step 6: Plot
x_vals = np.arange(len(df))
width = 0.25

plt.figure(figsize=(14, 7))
plt.bar(x_vals - width, temperature, width, label='Temperature (°C)', color='tomato')
plt.bar(x_vals, humidity, width, label='Humidity (%)', color='skyblue')
plt.bar(x_vals + width, aqi, width, label='Air Quality Index (AQI)', color='lightgreen')

# Regression line
plt.plot(x_vals, temperature_pred, color='red', linestyle='--', linewidth=2, label='Temp Trend (Regression)')

# Display average global warming
plt.text(len(x_vals) - 1, max(temperature) + 2,
         f'Avg. Global Warming: {average_gw_percent:.2f}%',
         color='darkred', weight='bold', fontsize=12)

# ✅ Clean Y-axis to remove "x0000D" mess
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Force integer ticks
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))  # Show as plain numbers

# Axis labels
plt.xticks(x_vals, x_labels, rotation=45)
plt.xlabel('Entry ID - Area')
plt.ylabel('Value')
plt.title('Environmental Parameters by Area with Global Warming Percentage')
plt.legend()
plt.tight_layout()
plt.show()
