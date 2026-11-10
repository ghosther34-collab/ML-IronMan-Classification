import openml
import pandas as pd

# Mengambil dataset Scene dari OpenML
dataset = openml.datasets.get_dataset(40595)

# Mengambil seluruh data tanpa memisahkan target
df, _, _, _ = dataset.get_data(dataset_format="dataframe")

# Menyimpan dataset
df.to_csv("Dataset/scene.csv", index=False)

print("=== DOWNLOAD DATASET SCENE MULTILABEL ===")
print("Dataset berhasil diunduh!")

print("\nUKURAN DATASET:")
print(df.shape)

print("\nNAMA KOLOM:")
print(df.columns.tolist())

print("\nJUMLAH NILAI KOSONG:")
print(df.isnull().sum().sum())

print("\nINFORMASI DATASET:")
print(dataset.description[:500])
