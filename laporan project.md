
## INFORMASI PROYEK
Judul Proyek: Klasifikasi Jenis Kismis Menggunakan Algoritma Machine Learning

Nama Mahasiswa: Zihan Cahya Amelia

NIM: 233307030

Program Studi: Teknologi Informasi

Mata Kuliah: Data Science

Dosen Pengampu: Gus Nanang Syaifuddiin, S.Kom., M.Kom.

Tahun Akademik: 2025 / 5

Link GitHub Repository: (https://github.com/zihancahya30/DataScienceUAS.git)

Link Video Pembahasan: (BELUM)

---
## 1. LEARNING OUTCOMES
Pada proyek ini, mahasiswa diharapkan dapat:
1. Memahami konteks masalah dan merumuskan problem statement secara jelas
2. Melakukan analisis dan eksplorasi data (EDA) secara komprehensif (**OPSIONAL**)
3. Melakukan data preparation yang sesuai dengan karakteristik dataset
4. Mengembangkan tiga model machine learning yang terdiri dari (**WAJIB**):
   - Model baseline
   - Model machine learning / advanced
   - Model deep learning (**WAJIB**)
5. Menggunakan metrik evaluasi yang relevan dengan jenis tugas ML
6. Melaporkan hasil eksperimen secara ilmiah dan sistematis
7. Mengunggah seluruh kode proyek ke GitHub (**WAJIB**)
8. Menerapkan prinsip software engineering dalam pengembangan proyek
---
## 2. PROJECT OVERVIEW
### 2.1 Latar Belakang
Sektor pertanian dan industri makanan sangat bergantung pada kualitas produk untuk daya saing pasar. Dalam konteks produk pertanian olahan seperti kismis, identifikasi varietas secara akurat menjadi krusial untuk standardisasi kualitas, penentuan harga, dan efisiensi proses produksi. Secara tradisional, klasifikasi varietas kismis, seperti Besni dan Kecimen, seringkali dilakukan secara manual oleh tenaga ahli. Namun, metode ini rentan terhadap kesalahan manusia, tidak konsisten, dan sangat memakan waktu, terutama dalam skala produksi besar.

Pemanfaatan teknologi Machine Learning (ML) dan Deep Learning (DL) menawarkan solusi inovatif untuk mengatasi tantangan ini. Dengan menganalisis fitur-fitur geometris dan morfologi dari citra kismis, sistem otomatis dapat dikembangkan untuk mengidentifikasi varietas dengan cepat dan akurat. Pendekatan ini dapat meminimalisir bias, meningkatkan efisiensi operasional, dan memastikan kualitas produk yang lebih konsisten di pasar.

Beberapa studi ilmiah telah menunjukkan potensi besar ML dan DL dalam klasifikasi varietas kismis. Penelitian oleh Ramadhan, A. J., et al. (2023) dalam "*Deep Learning Model for Raisin Grains Classification*" berhasil menunjukkan bahwa model Deep Learning berbasis Convolutional Neural Network (CNN) dapat mencapai akurasi yang superior dibandingkan metode tradisional dalam klasifikasi kismis. Studi lain oleh Ramadhan, A. J., et al. (2023) juga menegaskan efektivitas algoritma Machine Learning klasik seperti Random Forest dan Logistic Regression untuk data geometris kismis.

Proyek ini memanfaatkan *Raisin Dataset* dari UCI Machine Learning Repository yang menyediakan fitur-fitur morfologi terukur dari dua varietas kismis (Besni dan Kecimen). Dengan menerapkan dan membandingkan model-model ML (Logistic Regression, Random Forest) dan DL (Multilayer Perceptron), proyek ini bertujuan untuk membangun sistem klasifikasi yang robust, efisien, dan akurat. Hasil dari proyek ini diharapkan tidak hanya memberikan kontribusi pada pengembangan metodologi klasifikasi pertanian berbasis AI, tetapi juga dapat menjadi landasan bagi implementasi solusi praktis di industri pengolahan kismis.

**Referensi Ilmiah:**
> Ramadhan, A. J., Al-Jumaili, S. A. A., & Ali, A. H. (2023). *Deep Learning Model for Raisin Grains Classification*. Journal of Theoretical and Applied Information Technology, 101(21). [Open Access Link](http://www.jatit.org/volumes/Vol101No21/31Vol101No21.pdf)
> Ramadhan, A. J., et al. (2023). *Classification of raisin grains variety using some machine learning methods*. ResearchGate. [Open Access Link](https://www.researchgate.net/publication/371204363_CLASSIFICATION_OF_RAISIN_GRAINS_VARIETY_USING_SOME_MACHINE_LEARNING_METHODS)
---
## 3. BUSINESS UNDERSTANDING / PROBLEM UNDERSTANDING
### 3.1 Problem Statements
1. Identifikasi varietas kismis Besni dan Kecimen secara manual sulit karena kemiripan fitur geometris.
2. Dataset memiliki fitur numerik dari citra yang memerlukan analisis ML untuk pola non-linear.
3. Diperlukan model untuk akurasi tinggi dalam klasifikasi biner.
4. Perbandingan baseline, advanced ML, dan deep learning untuk pendekatan optimal.
### 3.2 Goals
1. Bangun model klasifikasi varietas kismis dengan akurasi >85%.
2. Bandingkan tiga model: baseline, advanced, deep learning.
3. Evaluasi menggunakan metrik klasifikasi relevan.
4. Hasilkan sistem reproducible.
### 3.3 Solution Approach
#### **Model 1 – Baseline Model**
Logistic Regression dipilih karena sederhana, cepat, dan baik untuk baseline klasifikasi biner. Alasan: Memberi gambaran dasar performa linier.
#### **Model 2 – Advanced / ML Model**
Random Forest dipilih karena ensemble tree menangani non-linearitas dan robust. Alasan: Cocok untuk data tabular dengan fitur berkorelasi.
#### **Model 3 – Deep Learning Model (WAJIB)**
Multilayer Perceptron (MLP) untuk data tabular. Alasan: Mampu belajar fitur kompleks. Dilatih 50 epochs dengan plot loss/accuracy, evaluasi test set.
---
## 4. DATA UNDERSTANDING
### 4.1 Informasi Dataset
**Sumber Dataset:** UCI ML Repository (https://archive.ics.uci.edu/dataset/850/raisin)
**Deskripsi Dataset:**
- Jumlah baris (rows): 900
- Jumlah kolom (columns/features): 8
- Tipe data: Tabular
- Ukuran dataset: ~0.1 MB
- Format file: XLSX
### 4.2 Deskripsi Fitur
| Nama Fitur | Tipe Data | Deskripsi | Contoh Nilai |
|------------|-----------|-----------|--------------|
| Area | Integer | Luas area kismis | 87524, 143234 |
| MajorAxisLength | Float | Panjang sumbu utama kismis | 442.246, 728.927 |
| MinorAxisLength | Float | Panjang sumbu minor kismis | 253.291, 225.629 |
| Eccentricity | Float | Eksentrisitas | 0.819, 0.951 |
| ConvexArea | Integer | Luas convex | 90546, 206257 |
| Extent | Float | Rasio extent | 0.758, 0.586 |
| Perimeter | Float | Panjang keliling | 1184.040, 2369.731 |
| Class | Categorical | Label varietas | Besni, Kecimen |
### 4.3 Kondisi Data
- **Missing Values:** Tidak ada
- **Duplicate Data:** Tidak ada
- **Outliers:** Tidak signifikan
- **Imbalanced Data:** Seimbang (450:450)
- **Noise:** Rendah
- **Data Quality Issues:** Baik
### 4.4 Exploratory Data Analysis (EDA) - (**OPSIONAL**)
#### Visualisasi 1: Class Distribution
![Class Distribution](/images/class_distribution.png)
**Insight:** Kelas seimbang 50:50.
#### Visualisasi 2: Feature Correlation Heatmap
![Correlation Heatmap](/images/correlation_heatmap.png)
**Insight:** Korelasi tinggi antar fitur ukuran.
#### Visualisasi 3: Numerical Feature Distribution
![Histogram](/images/numerical_feature_distribution.png)
**Insight:** Distribusi normal dengan sedikit skew.
---
## 5. DATA PREPARATION
### 5.1 Data Cleaning
Tidak ada missing/duplikat. Konversi tipe jika perlu.
### 5.2 Feature Engineering
Tidak dilakukan, fitur asli cukup.
### 5.3 Data Transformation
Encoding: LabelEncoder untuk Class (Besni=0, Kecimen=1). Scaling: StandardScaler.
### 5.4 Data Splitting
- Training: 80% (720 samples)
- Test: 20% (180 samples)
Stratified, random_state=42.
### 5.5 Data Balancing (jika diperlukan)
Tidak perlu, kelas seimbang.
### 5.6 Ringkasan Data Preparation
1. **Apa:** Encoding dan scaling.
   **Mengapa:** Untuk input numerik dan skala seragam.
   **Bagaimana:** LabelEncoder dan StandardScaler.
2. **Apa:** Splitting.
   **Mengapa:** Evaluasi generalisasi.
   **Bagaimana:** train_test_split stratified.
---
## 6. MODELING
### 6.1 Model 1 — Baseline Model
#### 6.1.1 Deskripsi Model
**Nama Model:** Logistic Regression
**Teori Singkat:** Model linier untuk probabilitas kelas.
**Alasan Pemilihan:** Baseline sederhana.
#### 6.1.2 Hyperparameter
- max_iter: 1000
- random_state: 42
#### 6.1.3 Implementasi (Ringkas)
```python
from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression(random_state=42, max_iter=1000)
model_lr.fit(X_train_scaled, y_train)
```
### 6.1.4 Hasil Awal
Akurasi ~0.89
### 6.2 Model 2 — ML / Advanced Model
### 6.2.1 Deskripsi Model
Nama Model: Random Forest
Teori Singkat: Ensemble decision trees.
Alasan Pemilihan: Tangani non-linear.
Keunggulan: Robust.
Kelemahan: Lebih lambat.
### 6.2.2 Hyperparameter
- n_estimators: 100
- max_depth: 10
- random_state: 42

### 6.2.3 Implementasi (Ringkas)
```python
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model_rf.fit(X_train_scaled, y_train)
```
### 6.3 Model 3 — Deep Learning Model (WAJIB)
### 6.3.1 Deskripsi Model
Nama Model: MLP
** (Centang) Jenis Deep Learning: **
 Multilayer Perceptron (MLP) - untuk tabular
Alasan Pemilihan: Cocok tabular.
### 6.3.2 Arsitektur Model
- Dense 128 ReLU
- Dropout 0.3
- Dense 64 ReLU
- Dropout 0.3
- Dense 32 ReLU
- Dropout 0.2
- Dense 1 Sigmoid
### 6.3.3 Input & Preprocessing Khusus
Input shape: (7,)
Preprocessing: Scaling.
### 6.3.4 Hyperparameter
- Optimizer: Adam lr=0.001
- Loss: binary_crossentropy
- Metrics: accuracy
- Batch size: 32
- Epochs: 50
- Callbacks: EarlyStopping
### 6.3.5 Implementasi (Ringkas)
Framework: TensorFlow/Keras
```python
from tensorflow import keras
from tensorflow.keras import layers, callbacks

model_mlp = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])
model_mlp.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
history = model_mlp.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=0
)
```
### 6.3.6 Training Process

Training Time: ~1 menit
Computational Resource: Google Colab CPU
Training History Visualization:

![Training History](/images/history_loss_accuracy.png)

Analisis Training: Tidak overfitting, converge baik.

### 6.3.7 Model Summary
Model memiliki ~parameters sesuai arsitektur.
---
## 7. EVALUATION
### 7.1 Metrik Evaluasi

Accuracy, Precision, Recall, F1-Score, Confusion Matrix.

### 7.2 Hasil Evaluasi Model
#### 7.2.1 Model 1 (Baseline)

Metrik:

Accuracy: 0.8889

Precision: 0.8938

Recall: 0.8889

F1-Score: 0.8885
Confusion Matrix / Visualization:

![Confusion Matrix Logistic Regression](/images/confusion_matrix_logistic_regression.png)

#### 7.2.2 Model 2 (Advanced/ML)

Metrik:

Accuracy: 0.8556

Precision: 0.8704

Recall: 0.8556

F1-Score: 0.8541
Confusion Matrix / Visualization:

![Confusion Matrix Random Forest](/images/confusion_matrix_random_forest.png)

Feature Importance (jika applicable):

![Feature Importance](/images/feature_importance.png)

#### 7.2.3 Model 3 (Deep Learning)

Metrik:

Accuracy: 0.8778

Precision: 0.8846

Recall: 0.8778

F1-Score: 0.8772
Confusion Matrix / Visualization:

![Confusion Matrix Deep Learning](/images/confusion_matrix_deep_learning.png)

Training History:
[Sudah diinsert di Section 6.3.6]

### 7.3 Perbandingan Ketiga Model

Tabel Perbandingan:

| Model | Accuracy | Precision | Recall | F1-Score | Training Time | Inference Time |
|-------|----------|-----------|--------|----------|---------------|----------------|
| Baseline (Model 1) | 0.8889 | 0.8938 | 0.8889 | 0.8885 | 0.1s | 0.001s |
| Advanced (Model 2) | 0.8556 | 0.8704 | 0.8556 | 0.8541 | 1s | 0.05s |
| Deep Learning (Model 3) | 0.8778 | 0.8846 | 0.8778 | 0.8772 | 60s | 0.1s |
Visualisasi Perbandingan:

![Model Performance Comparison](/images/model_performance_comparison.png)

### 7.4 Analisis Hasil

Model Terbaik: Logistic Regression

Perbandingan dengan Baseline: Tidak ada peningkatan signifikan dari model lain.

Trade-off: Baseline cepat dan akurat.

Error Analysis: Kesalahan pada fitur mirip.

Overfitting/Underfitting: Tidak ada.

---
## 8. CONCLUSION
### 8.1 Kesimpulan Utama

Model Terbaik: Logistic Regression
Alasan: Akurasi tertinggi.
Pencapaian Goals: Tercapai.

### 8.2 Key Insights

Insight dari Data: Fitur berkorelasi tinggi.
Insight dari Modeling: Model sederhana cukup.

### 8.3 Kontribusi Proyek
Manfaat praktis: Otomatisasi klasifikasi kismis.
Pembelajaran yang didapat: Alur ML lengkap.
---
## 9. FUTURE WORK (Opsional)

Berikut adalah daftar potensi pekerjaan di masa mendatang dalam format checklist:

- [x] **Tambah data:** Mengumpulkan lebih banyak data kismis dari varietas yang berbeda atau dalam kondisi yang berbeda untuk meningkatkan generalisasi model.
- [x] **Menambah variasi data:** (Dipertimbangkan sebagai bagian dari 'Tambah data')
- [x] **Feature engineering lebih lanjut:** Membuat fitur baru dari fitur yang ada atau mengeksplorasi teknik ekstraksi fitur lainnya.

**Model:**
- [x] **Mencoba arsitektur DL yang lebih kompleks:** Mengeksplorasi model DL seperti CNN atau model yang lebih dalam/lebar.
- [x] **Hyperparameter tuning lebih ekstensif:** Melakukan optimasi hyperparameter yang lebih mendalam untuk semua model (misalnya dengan GridSearchCV, RandomizedSearchCV, atau Keras Tuner).
- [x] **Ensemble methods (combining models):** Menggabungkan beberapa model untuk meningkatkan kinerja secara keseluruhan.
- [x] **Transfer learning dengan model yang lebih besar:** (Kurang relevan untuk data tabular ini)

**Deployment:**
- [x] **Membuat API (Flask/FastAPI):** Membuat antarmuka pemrograman aplikasi untuk model agar dapat diakses oleh aplikasi lain.
- [x] **Membuat web application (Streamlit/Gradio):** Membuat aplikasi web interaktif untuk mendemonstrasikan model.
- [x] **Containerization dengan Docker:** Mengemas aplikasi dan semua dependensinya ke dalam container untuk deployment yang konsisten.
- [x] **Deploy ke cloud (Heroku, GCP, AWS):** Menyebarkan aplikasi ke platform cloud untuk akses publik atau skalabilitas.

**Optimization:**
- [x] **Model compression (pruning, quantization):** Mengurangi ukuran dan kompleksitas model untuk inferensi yang lebih cepat dan penggunaan memori yang lebih rendah.
- [x] **Improving inference speed:** Mengoptimalkan waktu yang dibutuhkan model untuk membuat prediksi.
- [x] **Reducing model size:** Mengurangi ukuran file model tanpa mengurangi akurasi secara signifikan.

---
## 10. REPRODUCIBILITY (WAJIB)
### 10.1 GitHub Repository

Link Repository: (https://github.com/zihancahya30/DataScienceUAS.git)
Repository harus berisi:

- [x] Notebook Jupyter/Colab dengan hasil running
- [x] Script Python (jika ada)
- [x] `requirements.txt` atau `environment.yml`
- [x] `README.md` yang informatif
- [x] Folder structure yang terorganisir
- [ ] `.gitignore` (jangan upload dataset besar)

### 10.2 Environment & Dependencies

Python Version: 3.10
Main Libraries & Versions:

numpy==1.23.5
pandas==1.5.3
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.12.2
tensorflow==2.12.0
joblib==1.2.0
ucimlrepo==0.0.3


