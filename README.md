
# 📘 Judul Proyek
*(Klasifikasi Jenis Kismis Menggunakan Algoritma Machine Learning)*

## 👤 Informasi
- **Nama:** [Zihan Cahya Amelia]
- **Repo:** [https://github.com/zihancahya30/DataScienceUAS.git]
- **Video:** [Link Video Presentasi/Demo]

---

# 1. 🎯 Ringkasan Proyek
- Menyelesaikan permasalahan klasifikasi jenis kismis (Besni atau Kecimen).
- Melakukan data preparation meliputi encoding variabel target, pembagian data, dan scaling fitur numerik.
- Membangun 3 model: **Logistic Regression (Baseline)**, **Random Forest (Advanced)**, dan **Deep Learning (MLP)**.
- Melakukan evaluasi performa model dan menentukan model terbaik berdasarkan metrik yang digunakan.

---

# 2. 📄 Problem & Goals
**Problem Statements:**
- Bagaimana membangun model klasifikasi yang efektif untuk membedakan antara dua jenis kismis (Besni dan Kecimen) berdasarkan fitur-fitur geometrisnya?
- Model mana di antara Logistic Regression, Random Forest, dan Deep Learning (MLP) yang memberikan performa terbaik untuk tugas klasifikasi ini?

**Goals:**
- Mengembangkan model Machine Learning dan Deep Learning untuk klasifikasi kismis.
- Menganalisis dan membandingkan performa model-model yang dibangun.
- Mengidentifikasi fitur-fitur penting yang memengaruhi klasifikasi kismis.
- Menyimpan model terbaik untuk potensi penggunaan di masa depan.

---
## 📁 Struktur Folder
```
project/
│
├── data/                   # Dataset (tidak di-commit, download manual)
│   └── Raisin_Dataset.xlsx
│
├── notebooks/              # Jupyter notebooks
│   └── 233307030_UAS_DATA SCIENCE_RAISIN DATASET.ipynb # Contoh nama notebook ini
│
├── src/                    # Source code
│   ├── load_data.py
│   ├── data_preparation.py
│   ├── eda_class_distribution.py
│   ├── eda_correlation_heatmap.py
│   ├── eda_numerical_feature_distribution.py
│   ├── model_evaluation_function.py
│   ├── train_logistic_regression.py
│   ├── train_random_forest.py
│   ├── train_deep_learning_mlp.py
│   ├── model_comparison.py
│   ├── save_models.py
│   └── main.py
│
├── models/                 # Saved models
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── deep_learning_mlp_model.h5
│
├── images/                 # Visualizations and plots
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── numerical_feature_distribution.png
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_random_forest.png
│   ├── confusion_matrix_deep_learning.png
│   ├── feature_importance.png
│   ├── history_loss_accuracy.png
│   └── model_performance_comparison.png
│
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore file
└── README.md               # Project README file
```
---

# 3. 📊 Dataset
- **Sumber:** UCI Machine Learning Repository - Raisin Dataset ([https://archive.ics.uci.edu/dataset/850/raisin](https://archive.ics.uci.edu/dataset/850/raisin))
- **Jumlah Data:** 900 baris (instance), 8 kolom (fitur + target)
- **Tipe:** Data numerik kontinu, masalah klasifikasi biner.

### Fitur Utama
| Fitur | Deskripsi |
|------|-----------|
| Area | Luas area dari kismis |
| MajorAxisLength | Panjang sumbu utama kismis |
| MinorAxisLength | Panjang sumbu minor kismis |
| Eccentricity | Eksentrisitas bentuk elips kismis |
| ConvexArea | Luas area cembung kismis |
| Extent | Rasio piksel dalam bounding box dengan total piksel area |
| Perimeter | Panjang keliling kismis |
| Class | Kelas kismis (Besni atau Kecimen) - *Target Variable* |

---

# 4. 🔧 Data Preparation
- **Cleaning:** Tidak ditemukan missing values atau duplikasi. Outliers tidak ditangani secara eksplisit dalam tahap ini karena model yang digunakan cukup robust atau scaling sudah cukup membantu.
- **Transformasi:** Target variable `Class` di-encode menjadi numerik (Besni=0, Kecimen=1) menggunakan `LabelEncoder`. Fitur numerik distandarisasi menggunakan `StandardScaler`.
- **Splitting:** Data dibagi menjadi 80% data training dan 20% data testing menggunakan `train_test_split` dengan `random_state=42` dan `stratify=y` untuk menjaga proporsi kelas.

---

# 5. 🤖 Modeling
- **Model 1 – Baseline:** Logistic Regression.
    - Diimplementasikan menggunakan `sklearn.linear_model.LogisticRegression`.
    - Digunakan sebagai dasar perbandingan karena kesederhanaan dan interpretabilitasnya.
- **Model 2 – Advanced ML:** Random Forest.
    - Diimplementasikan menggunakan `sklearn.ensemble.RandomForestClassifier`.
    - Menggunakan 100 estimator dan `max_depth=10` untuk keseimbangan performa dan overfitting.
    - Juga menganalisis *feature importance* untuk mengidentifikasi fitur paling berpengaruh.
- **Model 3 – Deep Learning:** Multilayer Perceptron (MLP).
    - Dibangun menggunakan `tensorflow.keras.Sequential`.
    - Terdiri dari beberapa layer `Dense` dengan aktivasi `relu` dan layer `Dropout` untuk regularisasi.
    - Menggunakan `sigmoid` di output layer untuk klasifikasi biner dan di-compile dengan optimizer Adam serta `loss='binary_crossentropy'`.

---

# 6. 🧪 Evaluation
**Metrik:** Accuracy, Precision, Recall, F1-Score (untuk klasifikasi biner)

### Hasil Singkat
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.8889   | 0.8938    | 0.8889 | 0.8885   |
| Random Forest | 0.8556   | 0.8704    | 0.8556 | 0.8541   |
| Deep Learning (MLP) | 0.8833   | 0.8891    | 0.8833 | 0.8829   |

---

# 7. 🏁 Kesimpulan
- **Model terbaik:** Logistic Regression
- **Alasan:** Berdasarkan metrik Accuracy, Precision, Recall, dan F1-Score, model Logistic Regression menunjukkan performa sedikit lebih tinggi dibandingkan dengan Random Forest dan Deep Learning (MLP) pada dataset ini.
- **Insight penting:** Meskipun Logistic Regression adalah model yang lebih sederhana, ia mampu mencapai performa yang sangat baik. Fitur-fitur geometris kismis ternyata memiliki korelasi yang kuat dan cukup linear untuk diklasifikasikan secara efektif oleh model linear. Random Forest juga mengidentifikasi `Perimeter` dan `MajorAxisLength` sebagai fitur terpenting.

---

# 8. 🔮 Future Work
- [ ] **Tambah data:** Mengumpulkan lebih banyak data kismis dari varietas yang berbeda atau dalam kondisi yang berbeda untuk meningkatkan generalisasi model.
- [ ] **Tuning model:** Melakukan hyperparameter tuning yang lebih ekstensif untuk semua model (misalnya dengan GridSearchCV atau RandomizedSearchCV untuk ML, dan Keras Tuner untuk DL).
- [ ] **Coba arsitektur DL lain:** Mengeksplorasi arsitektur jaringan saraf yang lebih kompleks atau mencoba model pre-trained (jika relevan).
- [ ] **Deployment:** Mengembangkan aplikasi web sederhana atau API untuk menguji model secara real-time.

---

# 9. 🔁 Reproducibility
Gunakan `requirements.txt` untuk menginstal dependensi yang diperlukan. Pastikan Anda memiliki Python versi 3.x.x. Anda dapat menjalankan `main.py` di folder `src` untuk mereproduksi seluruh alur kerja:

```bash
pip install -r requirements.txt
python src/main.py
```
