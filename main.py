import pandas as pd

df = pd.read_csv("Dataset/all_ironman_suit.csv")

# Membersihkan spasi pada nama kolom
df.columns = df.columns.str.strip()

# Menampilkan 5 baris pertama
print("=== 5 DATA PERTAMA ===")
print(df.head())

# Menampilkan ukuran dataset
print("\n=== UKURAN DATASET ===")
print("Jumlah baris dan kolom:", df.shape)

# Menampilkan informasi setiap kolom
print("\n=== INFORMASI DATASET ===")
df.info()

# Menghitung jumlah data kosong
print("\n=== JUMLAH DATA KOSONG ===")
print(df.isnull().sum())

print("\n=== CONTOH ESTIMATED VALUE ===")
print(df["Estimated Value"].head(10).to_list())

print("\n=== CONTOH CODENAME ===")
print(df["Codename"].head(15).to_list())

# Memeriksa data kosong
print("\n=== DATA KOSONG ===")
print(df.isnull().sum())

# Memeriksa jumlah baris duplikat
print("\n=== DATA DUPLIKAT ===")
print("Jumlah baris duplikat:", df.duplicated().sum())

# Mengisi Codename yang kosong
df["Codename"] = df["Codename"].fillna("Unknown")

# Memeriksa kembali data kosong
print("\n=== DATA KOSONG SETELAH PEMBERSIHAN ===")
print(df.isnull().sum())

# Memeriksa hasil Codename
print("\n=== CONTOH CODENAME SETELAH PEMBERSIHAN ===")
print(df["Codename"].head(15).to_list())

# Mengubah Estimated Value menjadi angka dalam juta USD
def konversi_nilai(nilai):
    nilai = nilai.replace("$", "").strip()

    if "Billion" in nilai:
        return float(nilai.replace(" Billion", "")) * 1000
    elif "Million" in nilai:
        return float(nilai.replace(" Million", ""))
    else:
        return float(nilai)

df["Estimated Value (Million USD)"] = (
    df["Estimated Value"].apply(konversi_nilai)
)

# Memeriksa hasil konversi
print("\n=== NILAI ARMOR DALAM JUTA USD ===")
print(
    df[["Model", "Estimated Value",
        "Estimated Value (Million USD)"]].head(10)
)

# Statistik deskriptif nilai armor
print("\n=== STATISTIK NILAI ARMOR ===")

print(
    df["Estimated Value (Million USD)"].describe()
)

# Menampilkan nilai armor tertinggi
nilai_tertinggi = df["Estimated Value (Million USD)"].max()
print("\nNilai armor tertinggi:", nilai_tertinggi, "juta USD")

# Menampilkan nilai armor terendah
nilai_terendah = df["Estimated Value (Million USD)"].min()
print("Nilai armor terendah:", nilai_terendah, "juta USD")

# Menghitung rata-rata nilai armor
rata_rata = df["Estimated Value (Million USD)"].mean()
print("Rata-rata nilai armor:", round(rata_rata, 2), "juta USD")

import matplotlib.pyplot as plt

# Histogram nilai armor
plt.figure(figsize=(8, 5))

plt.hist(
    df["Estimated Value (Million USD)"],
    bins=10,
    edgecolor="black"
)

plt.title("Distribusi Nilai Armor Iron Man")
plt.xlabel("Estimated Value (Million USD)")
plt.ylabel("Jumlah Armor")
plt.tight_layout()
plt.show()

# Mengurutkan armor berdasarkan nilai tertinggi
print("\n=== 5 ARMOR DENGAN NILAI TERTINGGI ===")

armor_tertinggi = df.sort_values(
    by="Estimated Value (Million USD)",
    ascending=False
)

print(
    armor_tertinggi[
        ["Model", "Era", "Codename", "Estimated Value"]
    ].head(5).to_string(index=False)
)

# Rata-rata nilai armor berdasarkan Era
print("\n=== RATA-RATA NILAI ARMOR BERDASARKAN ERA ===")

rata_era = df.groupby("Era")[
    "Estimated Value (Million USD)"
].mean().sort_values(ascending=False)

print(rata_era.round(2))

# Grafik rata-rata nilai armor berdasarkan Era
plt.figure(figsize=(9, 5))

rata_era.plot(kind="bar", edgecolor="black")

plt.title("Rata-rata Nilai Armor Berdasarkan Era")
plt.xlabel("Era")
plt.ylabel("Rata-rata Nilai (Juta USD)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Menghitung jumlah armor berdasarkan Era
print("\n=== JUMLAH ARMOR BERDASARKAN ERA ===")

jumlah_era = df["Era"].value_counts()

print(jumlah_era)

# Menentukan fitur dan target klasifikasi
print("\n=== MENENTUKAN FITUR DAN TARGET ===")

fitur = [
    "Model",
    "Codename",
    "Primary Ability",
    "Date Made"
]

X = df[fitur]
y = df["Era"]

print("Fitur yang digunakan:")
print(X.head())

print("\nTarget yang diprediksi:")
print(y.head())

print("\nUkuran fitur:", X.shape)
print("Jumlah target:", y.shape[0])

print("\nDistribusi target:")
print(y.value_counts())

# Membagi data menjadi data latih dan data uji
from sklearn.model_selection import train_test_split

print("\n=== MEMBAGI DATA LATIH DAN DATA UJI ===")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Jumlah data latih:", len(X_train))
print("Jumlah data uji:", len(X_test))

print("\nDistribusi Era pada data latih:")
print(y_train.value_counts())

print("\nDistribusi Era pada data uji:")
print(y_test.value_counts())

# Membuat model klasifikasi Decision Tree
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

print("\n=== MEMBUAT MODEL KLASIFIKASI ===")

# Mengambil tahun dari kolom Date Made
import re

df["Year Made"] = df["Date Made"].str.extract(r"(\d{4})").astype(int)

# Memilih fitur yang digunakan
fitur_model = ["Codename", "Primary Ability", "Year Made"]
X_model = df[fitur_model]
y_model = df["Era"]

y_model = df["Era"]

# Membagi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(
    X_model,
    y_model,
    test_size=0.25,
    random_state=42,
    stratify=y_model
)

# Memisahkan fitur teks dan angka
fitur_teks = ["Codename", "Primary Ability"]
fitur_angka = ["Year Made"]

# Mengatur preprocessing setiap fitur
preprocessor = ColumnTransformer(
    transformers=[
        ("teks", OneHotEncoder(handle_unknown="ignore"), fitur_teks),
        ("angka", "passthrough", fitur_angka)
    ]
)

# Membuat pipeline agar pengolahan data dan model terhubung

model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42
    ))
])

# Melatih model
model.fit(X_train, y_train)

# === MENYIMPAN MODEL ===
import joblib

joblib.dump(model, "model_ironman.pkl")

print("Model berhasil disimpan: model_ironman.pkl")

# === MEMUAT MODEL TERSIMPAN ===
model_tersimpan = joblib.load("model_ironman.pkl")

print("Model berhasil dimuat kembali.")

# Memeriksa kedalaman pohon
print("Kedalaman maksimum yang diatur:",
      model.named_steps["classifier"].max_depth)

print("Kedalaman pohon yang terbentuk:",
      model.named_steps["classifier"].get_depth())

# Mengevaluasi model klasifikasi
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\n=== EVALUASI MODEL ===")

# Melakukan prediksi pada data uji
y_pred = model.predict(X_test)

# Menghitung akurasi
akurasi = accuracy_score(y_test, y_pred)
print("Akurasi model:", round(akurasi * 100, 2), "%")

# Menampilkan hasil evaluasi setiap kategori
print("\nLaporan klasifikasi:")
print(classification_report(y_test, y_pred, zero_division=0))

# Menampilkan confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
))
print("Urutan kategori:", model.classes_)

# Membandingkan hasil prediksi dengan kategori sebenarnya
print("\n=== PERBANDINGAN HASIL PREDIKSI ===")

hasil_prediksi = X_test.copy()
hasil_prediksi["Era Sebenarnya"] = y_test
hasil_prediksi["Era Prediksi"] = y_pred
hasil_prediksi["Status"] = (
    hasil_prediksi["Era Sebenarnya"]
    == hasil_prediksi["Era Prediksi"]
).map({True: "Benar", False: "Salah"})

print(hasil_prediksi.to_string())

# Memeriksa tanggal pembuatan berdasarkan Era
print("\n=== TANGGAL PEMBUATAN BERDASARKAN ERA ===")

print(
    df.groupby("Era")["Date Made"]
    .value_counts()
    .to_string()
)

# Mengambil tahun dari tanggal pembuatan
import re

print("\n=== MENGAMBIL TAHUN PEMBUATAN ===")

df["Year Made"] = df["Date Made"].str.extract(
    r"(\d{4})"
).astype(int)

print(df[["Model", "Era", "Date Made", "Year Made"]].to_string(index=False))

# === VISUALISASI POHON KEPUTUSAN ===
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Mengambil Decision Tree dari Pipeline
pohon = model.named_steps["classifier"]

# Membuat visualisasi
plt.figure(figsize=(22, 10))
plot_tree(
    pohon,
    feature_names=model.named_steps["preprocessing"].get_feature_names_out(),
    class_names=model.classes_,
    filled=True,
    rounded=True,
    fontsize=7
)

plt.title("Visualisasi Decision Tree Klasifikasi Era Iron Man")
plt.show()

# === PREDIKSI DATA BARU ===
print("\n=== PREDIKSI DATA BARU ===")

# Data baju Iron Man yang ingin diprediksi
data_baru = pd.DataFrame({
    "Codename": ["Unknown"],
    "Primary Ability": ["Advanced flight and energy weapons"],
    "Year Made": [2008]
})

# Melakukan prediksi menggunakan model yang sudah dilatih
hasil_prediksi = model.predict(data_baru)

print("Data yang diprediksi:")
print(data_baru)

print("Hasil prediksi era:", hasil_prediksi[0])



# === PREDIKSI BERULANG ===
print("\n=== PREDIKSI BERULANG ERA IRON MAN ===")

while True:
    codename = input("\nMasukkan Codename: ")
    primary_ability = input("Masukkan Primary Ability: ")

    try:
        year_made = int(input("Masukkan Tahun Pembuatan: "))
    except ValueError:
        print("Tahun harus berupa angka, misalnya 2008 atau 2018.")
        continue

    data_interaktif = pd.DataFrame({
        "Codename": [codename],
        "Primary Ability": [primary_ability],
        "Year Made": [year_made]
    })

    hasil = model_tersimpan.predict(data_interaktif)

    print("\n=== HASIL PREDIKSI ===")
    print("Codename:", codename)
    print("Tahun Pembuatan:", year_made)
    print("Prediksi Era:", hasil[0])

    ulang = input("\nPrediksi data lain? (y/n): ").lower()

    if ulang != "y":
        print("Program prediksi selesai.")
        break
