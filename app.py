import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Data
def load_data():
    df = pd.read_csv("C:/Users/Manvitha/Downloads/price/cars_data .csv")
    
    # Data Cleaning
    df['MSRP'] = df['MSRP'].astype(str).str.replace(r'[$,]', '', regex=True).str.strip()
    df['MSRP'] = pd.to_numeric(df['MSRP'], errors='coerce')
    df['Invoice'] = df['Invoice'].astype(str).str.replace(r'[$,]', '', regex=True).str.strip()
    df['Invoice'] = pd.to_numeric(df['Invoice'], errors='coerce')
    df['Cylinders'] = df['Cylinders'].fillna(df['Cylinders'].median())
    df = df.dropna()
    return df

# Train Model
def train_model(df):
    features = ["Invoice", "EngineSize", "Cylinders", "Horsepower", "MPG_City", 
                "MPG_Highway", "Weight", "Wheelbase", "Length"]
    target = "MSRP"
    
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred

# Streamlit Web App
st.title("Car Price Prediction Web App")

df = load_data()
model, X_test, y_test, y_pred = train_model(df)

# Show Performance Metrics
st.subheader("Model Performance")
st.write("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
st.write("R2 Score:", r2_score(y_test, y_pred))

# Scatter Plot of Actual vs Predicted Prices
st.subheader("Actual vs Predicted Car Prices")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.5, c='blue', label="Predicted Prices")
ax.set_xlabel("Actual Prices")
ax.set_ylabel("Predicted Prices")
ax.set_title("Actual vs. Predicted Car Prices")
ax.legend()
st.pyplot(fig)

# Display Best Cars
st.subheader("Best Cars Based on Prediction")
df_test = X_test.copy()
df_test["Actual Price"] = y_test
df_test["Predicted Price"] = y_pred
df_test = df_test.sort_values(by="Predicted Price", ascending=False)
st.dataframe(df_test.head(10))
