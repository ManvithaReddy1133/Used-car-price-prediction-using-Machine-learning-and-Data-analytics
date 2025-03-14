import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("C:/Users/Manvitha/Downloads/price/cars_data .csv")

# Data Cleaning
df['MSRP'] = df['MSRP'].astype(str).str.replace(r'[$,]', '', regex=True).str.strip()
df['MSRP'] = pd.to_numeric(df['MSRP'], errors='coerce')

df['Invoice'] = df['Invoice'].astype(str).str.replace(r'[$,]', '', regex=True).str.strip()
df['Invoice'] = pd.to_numeric(df['Invoice'], errors='coerce')

df['Cylinders'] = df['Cylinders'].fillna(df['Cylinders'].median())
df = df.dropna()

# Define features and target
features = ["Invoice", "EngineSize", "Cylinders", "Horsepower", "MPG_City", 
            "MPG_Highway", "Weight", "Wheelbase", "Length"]
target = "MSRP"

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model Performance
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 🔹 Step 1: Compare Actual vs. Predicted Prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, c='blue', label="Predicted Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted Car Prices")
plt.legend()
plt.show()

# 🔹 Step 2: Identify Best Cars Based on Predicted Price
df_test = X_test.copy()
df_test["Actual Price"] = y_test
df_test["Predicted Price"] = y_pred

best_cars = df_test.sort_values(by="Predicted Price", ascending=False)
print("\n🚗 Top 10 Best Cars Based on Predicted Price:")
print(best_cars.head(10))

# 🔹 Step 3: Identify Best Value-for-Money Cars
df_test["Value Score"] = df_test["Predicted Price"] / df_test["Invoice"]
best_value_cars = df_test.sort_values(by="Value Score", ascending=False)
print("\n💰 Top 10 Best Value-for-Money Cars:")
print(best_value_cars.head(10))

