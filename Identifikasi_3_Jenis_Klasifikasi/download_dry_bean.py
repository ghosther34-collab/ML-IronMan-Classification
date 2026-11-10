import pandas as pd
from ucimlrepo import fetch_ucirepo

# Mengambil dataset Dry Bean dari UCI
dry_bean = fetch_ucirepo(id=602)

# Menggabungkan fitur dan target
X = dry_bean.data.features
y = dry_bean.data.targets

df = pd.concat([X, y], axis=1)

# Menyimpan dataset
df.to_csv("Dataset/dry_bean.csv", index=False)

print("=== DOWNLOAD DATASET DRY BEAN ===")
print("Dataset berhasil diunduh!")

print("\nUKURAN DATASET:")
print(df.shape)

print("\nNAMA KOLOM:")
print(df.columns.tolist())

print("\nDISTRIBUSI TARGET:")
print(df.iloc[:, -1].value_counts())

print("\nJUMLAH KATEGORI TARGET:")
print(df.iloc[:, -1].nunique())

print("\nJUMLAH NILAI KOSONG:")
print(df.isnull().sum().sum())
