# Aplikasi Analisis Sentimen Ulasan Aplikasi

Aplikasi web interaktif untuk menganalisis sentimen dari ulasan aplikasi menggunakan machine learning. Aplikasi ini dibangun dengan Streamlit dan menyediakan dua model klasifikasi: Support Vector Machine (SVM) dan Logistic Regression.

## Fitur

- **Analisis Data**: Eksplorasi dataset dengan visualisasi dan statistik deskriptif
- **Preprocessing**: Pembersihan dan transformasi teks ulasan
- **Pemodelan**: Training dan evaluasi model SVM dan Logistic Regression
- **Prediksi Sentimen**: Klasifikasi ulasan baru menjadi positif, netral, atau negatif
- **Visualisasi**: Grafik distribusi sentimen, wordcloud, dan metrik performa model

## Persyaratan Sistem

- Python 3.12 atau lebih tinggi
- pip (Python package manager)
- Git

## Instalasi dan Setup

### 1. Clone Repository

```bash
git clone https://github.com/muhammadrizqyhidayah/projek_deeplearningg.git
cd projek_deeplearningg
```

### 2. Buat Virtual Environment

**Windows PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Setelah virtual environment aktif, install semua library yang dibutuhkan:

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data

Aplikasi akan otomatis mendownload data NLTK yang diperlukan saat pertama kali dijalankan. Jika ada masalah, jalankan:

```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

## Cara Menjalankan Aplikasi

1. Pastikan virtual environment sudah aktif (lihat langkah 2 di atas)
2. Jalankan aplikasi dengan perintah:

```bash
streamlit run app.py
```

3. Browser akan otomatis terbuka di `http://localhost:8501`
4. Jika tidak terbuka otomatis, buka browser dan akses URL tersebut

## Cara Menggunakan Aplikasi

### Halaman 1: Analisis Data
- Upload file CSV dengan kolom ulasan
- Lihat statistik dan visualisasi data
- Eksplorasi distribusi sentimen

### Halaman 2: Preprocessing
- Upload dataset untuk preprocessing
- Terapkan text cleaning, tokenization, dan stemming
- Download hasil preprocessing

### Halaman 3: Pemodelan
- Upload dataset yang sudah di-preprocessing
- Pilih model: SVM atau Logistic Regression
- Training model dan lihat metrik evaluasi (accuracy, precision, recall, F1-score)
- Download model terlatih

### Halaman 4: Prediksi Sentimen

**Prediksi Teks:**
- Ketik ulasan di text area
- Pilih model (SVM atau Logistic Regression)
- Klik tombol prediksi
- Lihat hasil: emoji sentimen, label, dan confidence score

**Prediksi dari File:**
- Upload file CSV dengan kolom ulasan
- Pilih model untuk prediksi
- Klik tombol prediksi
- Download hasil prediksi dengan kolom sentimen dan confidence

## Format Dataset

Dataset harus berupa file CSV dengan struktur:
- **Untuk Training**: Kolom `ulasan` dan `label` (positif/netral/negatif)
- **Untuk Prediksi**: Minimal kolom `ulasan`

Contoh:
```csv
ulasan,label
"Aplikasi bagus dan mudah digunakan",positif
"Biasa saja tidak istimewa",netral
"Aplikasi sering error dan lambat",negatif
```

## Batasan Upload

- Maksimal ukuran file: 1000 MB
- Untuk dataset besar (>10.000 baris), disarankan menggunakan sampling untuk performa optimal

## Model yang Tersimpan

Aplikasi ini sudah menyertakan model pre-trained di folder `model/`:
- `svm_model.pkl` - Model SVM
- `pipeline_best.pkl` - Model Logistic Regression
- `tfidf_vectorizer.pkl` - TF-IDF Vectorizer
- `label_encoder.pkl` - Label Encoder

## Deployment

Aplikasi ini juga tersedia secara online di Streamlit Cloud:
[Link akan tersedia setelah deployment]

## Troubleshooting

**Error saat install dependencies:**
- Pastikan Python versi 3.12 atau lebih tinggi
- Update pip: `python -m pip install --upgrade pip`

**Port sudah digunakan:**
- Gunakan port lain: `streamlit run app.py --server.port 8502`

**Memory error saat training:**
- Kurangi ukuran dataset dengan sampling
- Gunakan parameter model yang lebih sederhana

## Teknologi yang Digunakan

- **Streamlit**: Framework web app
- **Scikit-learn**: Machine learning models
- **NLTK**: Natural language processing
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualisasi
- **Sastrawi**: Indonesian stemmer

## Lisensi

Projek ini dibuat untuk keperluan pembelajaran Deep Learning.

## Kontak

Muhammad Rizqy Hidayah - [GitHub](https://github.com/muhammadrizqyhidayah)
