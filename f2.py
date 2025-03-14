import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset

df = pd.read_csv("C:/Users/Manvitha/Downloads/price/cars_data_cleaned.csv")

# 1. Check basic info
print(df.info())

# 2. Summary statistics
print(df.describe())

# 3. Check missing values
print(df.isnull().sum())

# 4. Visualize price distribution
plt.figure(figsize=(8,5))
sns.histplot(df['MSRP'], bins=30, kde=True)
plt.title("Distribution of Car Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Select only numerical columns
numeric_df = df.select_dtypes(include=['number'])

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

