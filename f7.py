import pandas as pd

# Load the protected file (Python can still read it)

df = pd.read_csv("C:/Users/Manvitha/Downloads/price/cars_data .csv")
print(df['MSRP'].dtype)  # Check current dtype
print(df['MSRP'].head(10))  # Display first 10 values


df['MSRP'] = df['MSRP'].astype(str)  # Ensure it's a string
df['MSRP'] = df['MSRP'].str.replace('[\$,]', '', regex=True)  # Remove '$' and ',' symbols
df['MSRP'] = df['MSRP'].str.strip()  # Remove leading/trailing spaces
df['MSRP'] = pd.to_numeric(df['MSRP'], errors='coerce')  # Convert to float


print(df['MSRP'].dtype)  # Should be float64
print(df['MSRP'].isnull().sum())  # Check if any NaN values exist
print(df['MSRP'].head(10))  # Confirm correct values

from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
features = ["MSRP", "Invoice", "EngineSize", "Cylinders", "Horsepower", 
            "MPG_City", "MPG_Highway", "Weight", "Wheelbase", "Length"]
target = "MSRP"

X = df[features]  # Independent variables
y = df[target]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, df['MSRP'], test_size=0.2, random_state=42)
print(X_train.dtypes)  # Ensure all features are numeric
print(y_train.dtype)   # Ensure target variable is numeric

print(X_train.isnull().sum())  # Check missing values in features
print(y_train.isnull().sum())  # Check missing values in target

print(X_train.head())
print(y_train.head())
