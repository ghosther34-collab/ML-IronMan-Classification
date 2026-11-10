import pandas as pd
from ucimlrepo import fetch_ucirepo

# Mengambil dataset Adult dari UCI Repository
adult = fetch_ucirepo(id=2)

# Menggabungkan fitur dan target
X = adult.data.features
y = adult.data.targets

df = pd.concat([X, y], axis=1)

# Menyamakan penulisan kategori target
df["income"] = df["income"].str.strip().str.rstrip(".")

# Menyimpan dataset
df.to_csv("Dataset/adult.csv", index=False)

print("=== DOWNLOAD DATASET ADULT ===")
print("Dataset berhasil diunduh!")

print("\nUKURAN DATASET:")
print(df.shape)

print("\nNAMA KOLOM:")
print(df.columns.tolist())

print("\nDISTRIBUSI TARGET:")
print(df["income"].value_counts())

print("\nJUMLAH NILAI KOSONG:")
print(df.isnull().sum().sum())
