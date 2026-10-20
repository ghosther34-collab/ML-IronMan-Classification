import pandas as pd
import joblib

# Memuat model tersimpan
model = joblib.load("model_ironman.pkl")

print("=== PREDIKSI INTERAKTIF IRON MAN ===")
print("Model berhasil dimuat!")

while True:
    codename = input("\nMasukkan Codename: ")
    primary_ability = input("Masukkan Primary Ability: ")

    try:
        year_made = int(input("Masukkan Tahun Pembuatan: "))
    except ValueError:
        print("Tahun harus berupa angka, misalnya 2008.")
        continue

    data_baru = pd.DataFrame({
        "Codename": [codename],
        "Primary Ability": [primary_ability],
        "Year Made": [year_made]
    })

    hasil = model.predict(data_baru)

    print("\n=== HASIL PREDIKSI ===")
    print("Codename:", codename)
    print("Primary Ability:", primary_ability)
    print("Tahun Pembuatan:", year_made)
    print("Prediksi Era:", hasil[0])

    ulang = input("\nPrediksi data lain? (y/n): ").lower()

    if ulang != "y":
        print("Program prediksi selesai.")
        break
