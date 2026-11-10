import pandas as pd

# Membaca dataset Scene
df = pd.read_csv("Dataset/scene.csv")

# Memisahkan fitur dan label
label_columns = [
    "Beach",
    "Sunset",
    "FallFoliage",
    "Field",
    "Mountain",
    "Urban"
]

feature_columns = [col for col in df.columns if col.startswith("Att")]

print("=== IDENTIFIKASI DATASET SCENE MULTILABEL ===")

# Menampilkan 5 data pertama
print("\n5 DATA PERTAMA:")
print(df.head())

# Ukuran dataset
print("\nUKURAN DATASET:")
print(df.shape)

# Jumlah fitur dan label
print("\nJUMLAH FITUR:", len(feature_columns))
print("JUMLAH KOLOM LABEL:", len(label_columns))

# Distribusi setiap label
print("\nDISTRIBUSI SETIAP LABEL:")
print(df[label_columns].sum())

# Jumlah label aktif
label_aktif = (df[label_columns].sum() > 0).sum()
print("\nJUMLAH LABEL AKTIF:", label_aktif)

# Distribusi jumlah label per data
jumlah_label = df[label_columns].sum(axis=1)

print("\nDISTRIBUSI JUMLAH LABEL PER DATA:")
print(jumlah_label.value_counts().sort_index())

# Nilai kosong
print("\nJUMLAH NILAI KOSONG:")
print(df.isnull().sum().sum())

# Kesimpulan
print("\nKESIMPULAN:")
print("Dataset Scene termasuk Multilabel Classification")
print("karena satu data dapat memiliki lebih dari satu label.")
