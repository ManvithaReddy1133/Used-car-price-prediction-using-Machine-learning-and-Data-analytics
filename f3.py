from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pandas as pd
df = pd.read_csv("C:/Users/Manvitha/Downloads/price/cars_data_cleaned.csv")
# 1. Handle Missing Values (Fill or Drop)
df = df.dropna()  # Dropping rows with missing values

# 2. Convert Categorical Variables
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for later use

# 3. Split the data
X = df.drop(columns=['MSRP'])  # Features (excluding target)
y = df['MSRP']  # Target (price)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print dataset shapes
print(f"Training Data Shape: {X_train.shape}")
print(f"Testing Data Shape: {X_test.shape}")


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 1. Initialize and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 2. Make Predictions
y_pred = model.predict(X_test)

# 3. Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"📌 Model Evaluation:")
print(f"🔹 Mean Absolute Error (MAE): {mae}")
print(f"🔹 Root Mean Squared Error (RMSE): {rmse}")
print(f"🔹 R² Score: {r2}")




from sklearn.ensemble import RandomForestRegressor

# 1. Initialize and Train the Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 2. Make Predictions
y_pred_rf = rf_model.predict(X_test)

# 3. Evaluate Model Performance
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print(f"📌 Random Forest Model Evaluation:")
print(f"🔹 Mean Absolute Error (MAE): {mae_rf}")
print(f"🔹 Root Mean Squared Error (RMSE): {rmse_rf}")
print(f"🔹 R² Score: {r2_rf}")


from sklearn.metrics import r2_score

models = {
    "Linear Regression": model,
    "Random Forest": rf_model
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"{name} R² Score:", r2_score(y_test, y_pred))
