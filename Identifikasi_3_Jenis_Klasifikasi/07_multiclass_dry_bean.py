import pandas as pd

# Membaca dataset Dry Bean
df = pd.read_csv("Dataset/dry_bean.csv")

print("=== IDENTIFIKASI DATASET DRY BEAN ===")

# Menampilkan 5 data pertama
print("\n5 DATA PERTAMA:")
print(df.head())

# Ukuran dataset
print("\nUKURAN DATASET:")
print(df.shape)

# Jumlah fitur dan target
print("\nJUMLAH FITUR:", len(df.columns) - 1)
print("JUMLAH TARGET:", 1)

# Distribusi target
print("\nDISTRIBUSI TARGET:")
print(df["Class"].value_counts())

# Jumlah kategori
print("\nJUMLAH KATEGORI TARGET:")
print(df["Class"].nunique())

# Nilai kosong
print("\nJUMLAH NILAI KOSONG:")
print(df.isnull().sum().sum())

# Kesimpulan
print("\nKESIMPULAN:")
print("Dataset Dry Bean termasuk Multiclass Classification")
print("karena memiliki satu target dengan tujuh kategori kelas.")
