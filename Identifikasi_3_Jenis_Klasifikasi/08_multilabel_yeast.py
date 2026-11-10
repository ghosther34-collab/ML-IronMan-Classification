import pandas as pd

# Membaca dataset Yeast
df = pd.read_csv("Dataset/yeast.csv")

# Memisahkan fitur dan label
label_columns = [col for col in df.columns if col.startswith("Class")]
feature_columns = [col for col in df.columns if col.startswith("Att")]

print("=== IDENTIFIKASI DATASET YEAST MULTILABEL ===")

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

# Jumlah label yang aktif
label_aktif = (df[label_columns].sum() > 0).sum()
print("\nJUMLAH LABEL AKTIF:", label_aktif)

# Jumlah label pada setiap data
jumlah_label = df[label_columns].sum(axis=1)

print("\nDISTRIBUSI JUMLAH LABEL PER DATA:")
print(jumlah_label.value_counts().sort_index())

# Nilai kosong
print("\nJUMLAH NILAI KOSONG:")
print(df.isnull().sum().sum())

# Kesimpulan
print("\nKESIMPULAN:")
print("Dataset Yeast termasuk Multilabel Classification")
print("karena satu data dapat memiliki lebih dari satu label.")
