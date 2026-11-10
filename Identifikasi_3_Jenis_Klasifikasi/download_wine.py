import pandas as pd
from ucimlrepo import fetch_ucirepo

# Mengambil dataset Wine dari UCI
wine = fetch_ucirepo(id=109)

# Menggabungkan fitur dan target
X = wine.data.features
y = wine.data.targets

df = pd.concat([X, y], axis=1)

# Menyimpan dataset
df.to_csv("Dataset/wine.csv", index=False)

print("=== DOWNLOAD DATASET WINE ===")
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
