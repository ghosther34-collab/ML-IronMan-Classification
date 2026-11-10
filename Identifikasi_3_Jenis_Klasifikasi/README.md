# Identifikasi 3 Jenis Klasifikasi Machine Learning

## Deskripsi Proyek

Proyek ini dibuat untuk memenuhi tugas mata kuliah Machine Learning, yaitu mengidentifikasi tiga jenis klasifikasi dalam pembelajaran mesin, yaitu Binary Classification, Multiclass Classification, dan Multilabel Classification.

Terdapat 9 dataset yang digunakan, masing-masing 3 dataset untuk setiap jenis klasifikasi.

## 1. Binary Classification

Binary Classification merupakan metode klasifikasi dengan satu target yang memiliki dua kategori.

| No. | Dataset | Jumlah Data | Target | Jumlah Kategori |
|---|---|---:|---|---:|
| 1 | Banknote Authentication | 1.372 | class | 2 |
| 2 | Adult | 48.842 | income | 2 |
| 3 | Bank Marketing | 45.211 | y | 2 |

**Sumber dataset:**
- Banknote Authentication: https://archive.ics.uci.edu/dataset/267/banknote+authentication
- Adult: https://archive.ics.uci.edu/dataset/2/adult
- Bank Marketing: https://archive.ics.uci.edu/dataset/222/bank+marketing

## 2. Multiclass Classification

Multiclass Classification merupakan metode klasifikasi dengan satu target yang memiliki lebih dari dua kategori.

| No. | Dataset | Jumlah Data | Target | Jumlah Kategori |
|---|---|---:|---|---:|
| 1 | Iris | 150 | class | 3 |
| 2 | Wine | 178 | class | 3 |
| 3 | Dry Bean | 13.611 | Class | 7 |

**Sumber dataset:**
- Iris: https://archive.ics.uci.edu/dataset/53/iris
- Wine: https://archive.ics.uci.edu/dataset/109/wine
- Dry Bean: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

## 3. Multilabel Classification

Multilabel Classification merupakan metode klasifikasi ketika satu data dapat memiliki lebih dari satu label sekaligus.

| No. | Dataset | Jumlah Data | Jumlah Fitur | Jumlah Label |
|---|---|---:|---:|---:|
| 1 | Emotions | 593 | 72 | 6 |
| 2 | Yeast | 2.417 | 103 | 14 |
| 3 | Scene | 2.407 | 294 | 6 |

**Sumber dataset:**
- Emotions: https://www.openml.org/d/41465
- Yeast: https://www.openml.org/d/41473
- Scene: https://www.openml.org/d/40595

## Struktur Folder

```text
Identifikasi_3_Jenis_Klasifikasi/
│
├── Dataset/
│   ├── data_banknote_authentication.txt
│   ├── adult.csv
│   ├── bank_marketing.csv
│   ├── iris.data
│   ├── wine.csv
│   ├── dry_bean.csv
│   ├── emotions.csv
│   ├── yeast.csv
│   └── scene.csv
│
├── 01_binary_banknote.py
├── 02_multiclass_iris.py
├── 03_multilabel_emotions.py
├── 04_binary_adult.py
├── 05_binary_bank_marketing.py
├── 06_multiclass_wine.py
├── 07_multiclass_dry_bean.py
├── 08_multilabel_yeast.py
├── 09_multilabel_scene.py
│
├── download_emotions.py
├── download_adult.py
├── download_bank_marketing.py
├── download_wine.py
├── download_dry_bean.py
├── download_yeast.py
└── download_scene.py
```

## Kesimpulan

Berdasarkan hasil identifikasi, terdapat 9 dataset yang terbagi menjadi 3 jenis klasifikasi:

1. Binary Classification: 3 dataset.
2. Multiclass Classification: 3 dataset.
3. Multilabel Classification: 3 dataset.

Setiap dataset memiliki karakteristik target yang berbeda sesuai dengan jenis klasifikasinya.

## Tools dan Library

- Python
- Pandas
- ucimlrepo
- OpenML