import pandas as pd

# Membaca dataset Emotions
df = pd.read_csv("Dataset/emotions.csv")

# Menentukan 6 kolom label
label_columns = [
    "amazed.suprised",
    "happy.pleased",
    "relaxing.calm",
    "quiet.still",
    "sad.lonely",
    "angry.aggresive"
]

print("=== IDENTIFIKASI MULTILABEL CLASSIFICATION ===")

# Menampilkan lima data pertama
print("\nLIMA DATA PERTAMA:")
print(df.head())

# Menampilkan ukuran dataset
print("\nUKURAN DATASET:")
print(df.shape)

# Menampilkan jumlah fitur dan label
feature_columns = [
    col for col in df.columns if col not in label_columns
]

print("\nJUMLAH FITUR:", len(feature_columns))
print("JUMLAH LABEL:", len(label_columns))

# Memeriksa tipe data label
print("\nTIPE DATA LABEL:")
print(df[label_columns].dtypes)

# Memeriksa nilai kosong
print("\nNILAI KOSONG:")
print(df.isnull().sum().sum())

# Menghitung jumlah data True dan False setiap label
print("\nDISTRIBUSI SETIAP LABEL:")
for label in label_columns:
    print(f"\n{label}:")
    print(df[label].value_counts())

# Menghitung jumlah label aktif pada setiap data
jumlah_label = df[label_columns].sum(axis=1)

print("\nJUMLAH LABEL PADA SETIAP DATA:")
print(jumlah_label.value_counts().sort_index())

# Menghitung data yang memiliki lebih dari satu label
data_multilabel = (jumlah_label > 1).sum()

print("\nDATA DENGAN LEBIH DARI SATU LABEL:")
print(data_multilabel)

# Kesimpulan
print("\nKESIMPULAN:")
print("Dataset Emotions termasuk Multilabel Classification.")
print("Karena satu data dapat memiliki lebih dari satu label sekaligus.")
