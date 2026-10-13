import pandas as pd
import numpy as np

df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding="latin1")

print("Dataset shape:", df.shape)
print(df.head())
print(df.dtypes)
print("Missing values:")
print(df.isna().sum().sort_values(ascending=False).head(20))
print("Duplicate rows:", df.duplicated().sum())

old_count = len(df)
df = df.drop_duplicates()
print("Duplicates removed:", old_count - len(df))

# Remove extra spaces from text columns
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].map(
        lambda x: x.strip() if isinstance(x, str) else x
    )

# Fill missing numeric values with median
for col in df.select_dtypes(include=np.number).columns:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# Fill missing text values with Unknown
for col in df.select_dtypes(include=["object", "category"]).columns:
    if df[col].isna().any():
        df[col] = df[col].fillna("Unknown")
date_col = "order date (DateOrders)"
if date_col in df.columns:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

shipping_col = "Days for shipping (real)"
if shipping_col in df.columns:
    df[shipping_col] = pd.to_numeric(df[shipping_col], errors="coerce")
shipping_col = "Days for shipping (real)"

if shipping_col in df.columns:
    values = pd.to_numeric(df[shipping_col], errors="coerce").dropna()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = values[(values < lower) | (values > upper)]
    print("Possible outliers:", len(outliers))
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Demonstration only; choose suitable measurement columns first
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

scaled_df = df.copy()
if numeric_cols:
    scaler = MinMaxScaler()
    scaled_df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown="ignore")
# Example use:
# encoded = encoder.fit_transform(df[["Shipping Mode"]])
# Check that "Shipping Mode" matches the exact column name in your file.
print("Final shape:", df.shape)
print("Missing cells remaining:", int(df.isna().sum().sum()))
print("Duplicate rows remaining:", int(df.duplicated().sum()))

df.to_csv("DataCoSupplyChainDataset_cleaned.csv", index=False)
print("Cleaned dataset saved successfully.")