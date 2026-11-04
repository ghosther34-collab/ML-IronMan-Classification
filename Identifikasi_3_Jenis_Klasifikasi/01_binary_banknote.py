
"""
IDENTIFIKASI DATASET BINARY CLASSIFICATION
Dataset: Banknote Authentication
Sumber: UCI Machine Learning Repository
URL: https://archive.ics.uci.edu/dataset/267/banknote+authentication

Target: class
Kategori: 0 dan 1
"""

import pandas as pd

# Membaca dataset menggunakan pandas
df = pd.read_csv(
    "Dataset/data_banknote_authentication.txt",
    header=None,
    names=[
        "variance",
        "skewness",
        "curtosis",
        "entropy",
        "class"
    ]
)

# 1. Menampilkan lima data pertama
print("=== LIMA DATA PERTAMA ===")
print(df.head())

# 2. Menampilkan ukuran dataset
print("\n=== UKURAN DATASET ===")
print(df.shape)

# 3. Menampilkan nama kolom
print("\n=== NAMA KOLOM ===")
print(df.columns.tolist())

# 4. Menampilkan informasi dataset
print("\n=== INFORMASI DATASET ===")
df.info()

# 5. Memeriksa missing value
print("\n=== MISSING VALUE ===")
print(df.isnull().sum())

# 6. Menghitung jumlah setiap kategori
print("\n=== DISTRIBUSI LABEL ===")
print(df["class"].value_counts().sort_index())

# 7. Menghitung jumlah kategori
print("\n=== JUMLAH KATEGORI ===")
print(df["class"].nunique())

# 8. Kesimpulan
print("\n=== KESIMPULAN ===")
print("Dataset termasuk Binary Classification.")
print("Memiliki satu kolom target yaitu class.")
print("Target memiliki dua kategori, yaitu 0 dan 1.")
