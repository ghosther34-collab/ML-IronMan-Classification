import pandas as pd
from ucimlrepo import fetch_ucirepo

# Mengambil dataset Bank Marketing dari UCI
bank = fetch_ucirepo(id=222)

# Menggabungkan fitur dan target
X = bank.data.features
y = bank.data.targets

df = pd.concat([X, y], axis=1)

# Menyimpan dataset
df.to_csv("Dataset/bank_marketing.csv", index=False)

print("=== DOWNLOAD DATASET BANK MARKETING ===")
print("Dataset berhasil diunduh!")

print("\nUKURAN DATASET:")
print(df.shape)

print("\nNAMA KOLOM:")
print(df.columns.tolist())

print("\nDISTRIBUSI TARGET:")
print(df["y"].value_counts())

print("\nJUMLAH NILAI KOSONG:")
print(df.isnull().sum().sum())

print("\nDETAIL NILAI KOSONG PER KOLOM:")
print(df.isnull().sum())