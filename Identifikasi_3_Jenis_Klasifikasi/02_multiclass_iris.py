
"""
IDENTIFIKASI DATASET MULTICLASS CLASSIFICATION
Dataset: Iris Flower
Sumber: UCI Machine Learning Repository
URL: https://archive.ics.uci.edu/dataset/53/iris

Target: class
Kategori:
1. Iris-setosa
2. Iris-versicolor
3. Iris-virginica
"""

import pandas as pd

# Membaca dataset Iris menggunakan pandas
df = pd.read_csv(
    "Dataset/iris.data",
    header=None,
    names=[
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
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
print(df["class"].value_counts())

# 7. Menghitung jumlah kategori
print("\n=== JUMLAH KATEGORI ===")
print(df["class"].nunique())

# 8. Kesimpulan
print("\n=== KESIMPULAN ===")
print("Dataset termasuk Multiclass Classification.")
print("Memiliki satu kolom target yaitu class.")
print("Target memiliki tiga kategori berbeda.")
