import pandas as pd


df = pd.read_csv("data/emails.csv")

# Show first rows
print(df.head())

# Dataset information
print("\nDataset Info")
print(df.info())

# Check missing values
print("\nMissing Values")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows")
print(df.duplicated().sum())

# Label distribution
print("\nLabel Distribution")
print(df["label"].value_counts())

# ==========================================
# Data Cleaning
# ==========================================

# Remove rows with missing values
df = df.dropna()

# Remove duplicate rows
df = df.drop_duplicates()

# Standardize labels
df["label"] = df["label"].str.lower().str.strip()

print(df.info())
print(df.head())