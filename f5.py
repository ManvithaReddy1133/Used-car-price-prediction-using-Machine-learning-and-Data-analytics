import pandas as pd
df = pd.read_csv("C:/Users/Manvitha/Downloads/price/cars_data_cleaned.csv")
import matplotlib.pyplot as plt
import seaborn as sns
brand_avg_hp = df.groupby('Brand')['Horsepower'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=brand_avg_hp.index, y=brand_avg_hp.values, palette="magma")
plt.xticks(rotation=45)
plt.title("Comparision of HP")
plt.xlabel("Brand")
plt.ylabel("Horsepower")
plt.show()




plt.figure(figsize=(8,5))
sns.countplot(x=df['Type'], order=df['Brand'].value_counts().index, palette="coolwarm")
plt.xticks(rotation=90)
plt.title("Distribution of Vehicle Types")
plt.xlabel("Type")
plt.ylabel("Count")
plt.show()
