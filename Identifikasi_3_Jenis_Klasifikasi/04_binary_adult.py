import pandas as pd

# IDENTIFIKASI DATASET BINARY CLASSIFICATION
# Dataset: Adult (Census Income)
# Sumber: UCI Machine Learning Repository
# URL: https://archive.ics.uci.edu/dataset/2/adult

# Membaca dataset
df = pd.read_csv("Dataset/adult.csv")

print("=== IDENTIFIKASI BINARY CLASSIFICATION ===")
print("Dataset: Adult (Census Income)")

# Menampilkan lima data pertama
print("\nLIMA DATA PERTAMA:")
print(df.head())

# Ukuran dataset
print("\nUKURAN DATASET:")
print(df.shape)

# Jumlah fitur dan target
print("\nJUMLAH FITUR:", len(df.columns) - 1)
print("JUMLAH TARGET: 1")

# Distribusi kategori target
print("\nDISTRIBUSI TARGET:")
print(df["income"].value_counts())

# Jumlah kategori
print("\nJUMLAH KATEGORI:")
print(df["income"].nunique())

# Nilai kosong
print("\nJUMLAH NILAI KOSONG:")
print(df.isnull().sum().sum())

# Kesimpulan
print("\nKESIMPULAN:")
print("Dataset Adult termasuk Binary Classification.")
print("Karena hanya memiliki dua kategori target, yaitu <=50K dan >50K.")
