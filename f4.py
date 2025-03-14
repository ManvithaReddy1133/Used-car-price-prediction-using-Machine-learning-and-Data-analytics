import pandas as pd
df = pd.read_csv("C:/Users/Manvitha/Downloads/price/cars_data_cleaned.csv")

import matplotlib.pyplot as plt
import seaborn as sns

# Group by brand and calculate the average price
brand_avg_price = df.groupby('Make')['MSRP'].mean().sort_values(ascending=False).head(10)

# Plot
plt.figure(figsize=(10,5))
sns.histplot(x=brand_avg_price.index, y=brand_avg_price.values, palette="viridis")
plt.xticks(rotation=45)
plt.title("Top 10 Most Expensive Car Brands")
plt.ylabel("Average Price ($)")
plt.xlabel("Brand")
plt.show()
