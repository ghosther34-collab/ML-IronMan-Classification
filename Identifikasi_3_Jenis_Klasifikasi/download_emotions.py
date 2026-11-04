
import openml
import pandas as pd

# Mengambil dataset Emotions dari OpenML
dataset = openml.datasets.get_dataset(41465)

# Mengambil seluruh kolom tanpa memisahkan target
# karena dataset ini memiliki 6 kolom label
df, _, _, _ = dataset.get_data(
    dataset_format="dataframe"
)

# Menyimpan dataset ke folder Dataset
df.to_csv("Dataset/emotions.csv", index=False)

print("=== DOWNLOAD DATASET EMOTIONS ===")
print("Dataset berhasil diunduh!")

print("\n=== UKURAN DATASET ===")
print(df.shape)

print("\n=== NAMA KOLOM ===")
print(df.columns.tolist())

print("\n=== LIMA DATA PERTAMA ===")
print(df.head())

print("\n=== JUMLAH KOLOM ===")
print(len(df.columns))
