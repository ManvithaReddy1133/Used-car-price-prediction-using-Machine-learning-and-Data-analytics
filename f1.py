import pandas as pd

# Load the protected file (Python can still read it)

df = pd.read_csv("C:/Users/Manvitha/Downloads/price/cars_data .csv")
print(df['Type'].value_counts())  # Change column name if needed

# Clean MSRP and Invoice columns
#df['MSRP'] = df['MSRP'].replace('[$,]', '', regex=True).astype(float)
#df['Invoice'] = df['Invoice'].replace('[$,]', '', regex=True).astype(float)

# Save the cleaned file
#cleaned_file_path = ("C:/Users/Manvitha/Downloads/price/cars_data_cleaned.csv")
#df.to_csv(cleaned_file_path, index=False)

# Show the path of the cleaned file
#cleaned_file_path
