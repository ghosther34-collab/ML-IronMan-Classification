import pandas as pd

# Membaca dataset Bank Marketing
df = pd.read_csv("Dataset/bank_marketing.csv")

print("=== IDENTIFIKASI DATASET BANK MARKETING ===")

# Menampilkan 5 data pertama
print("\n5 DATA PERTAMA:")
print(df.head())

# Informasi ukuran dataset
print("\nUKURAN DATASET:")
print(df.shape)

# Menghitung jumlah fitur dan target
print("\nJUMLAH FITUR:", len(df.columns) - 1)
print("JUMLAH TARGET:", 1)

# Distribusi kategori target
print("\nDISTRIBUSI TARGET:")
print(df["y"].value_counts())

# Jumlah kategori target
print("\nJUMLAH KATEGORI TARGET:")
print(df["y"].nunique())

# Memeriksa nilai kosong
print("\nJUMLAH NILAI KOSONG:")
print(df.isnull().sum().sum())

# Kesimpulan
print("\nKESIMPULAN:")
print("Dataset Bank Marketing termasuk Binary Classification")
print("karena memiliki satu target dengan dua kategori, yaitu no dan yes.")
