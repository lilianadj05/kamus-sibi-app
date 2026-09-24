# Kamus SIBI — Prediksi Isyarat (Model Transformer, Tanpa Augmentasi)

Aplikasi web Streamlit untuk mengenali kosakata Bahasa Isyarat Indonesia (SIBI)
dari video unggahan maupun webcam, menggunakan **Model 2** dari eksperimen kamu:
Transformer Encoder 2-Block, tanpa augmentasi, fitur geometri tangan 164-dimensi.

## Struktur Proyek

```
kamus-sibi-app/
├── app.py                     # Aplikasi utama Streamlit
├── requirements.txt           # Dependency Python
├── packages.txt               # Dependency sistem (apt) untuk Streamlit Cloud
├── .streamlit/config.toml     # Tema warna aplikasi
├── models/
│   ├── weights/                # ← LETAKKAN FILE *.weights.h5 KAMU DI SINI
│   │   └── Model_2_iter*_seed*.weights.h5
│   └── README.md
└── utils/
    ├── constants.py            # Parameter & daftar kelas
    ├── preprocessing.py        # Pipeline MediaPipe → 45×164 (identik dgn training)
    ├── model_utils.py          # Custom layer/loss + loader model
    └── ui.py                   # Styling (warna & font Calibri)
```

## Langkah 1 — Siapkan File Bobot Model

Bobot Model 2 (Transformer, Tanpa Augmentasi) kamu tersimpan per fold-iterasi
dan per seed, misalnya `Model_2_iter4_seed123.weights.h5`. Letakkan **semua**
file tersebut (atau sebagian yang mau dipakai) di folder `models/weights/`:

```
models/weights/Model_2_iter1_seed42.weights.h5
models/weights/Model_2_iter2_seed42.weights.h5
models/weights/Model_2_iter3_seed123.weights.h5
...
```

Aplikasi akan otomatis mendeteksi semua file `*.weights.h5` di folder ini,
membangun ulang arsitektur Transformer (`utils/model_utils.py`), memuat
bobotnya, lalu **meng-ensemble** (merata-ratakan) hasil prediksi dari fold/seed
yang dipilih. Secara default hanya file bertag `seed42` yang otomatis
terpilih (5 fold, 1 seed) — kamu bisa mengganti subset-nya dari UI aplikasi
(menu Prediksi → "Pengaturan model"). Kalau folder kamu berbeda, ubah
`WEIGHTS_DIR` di `utils/constants.py`.

## Mode Webcam

- Buffer webcam **fixed 6 detik terakhir** (tidak ada opsi durasi lain).
- Selama merekam, video akan menampilkan kotak hijau + tulisan
  "Tangan Terdeteksi"/"Tangan Tidak Terdeteksi" sebagai feedback langsung.
- Klik **"Prediksi & Hentikan Kamera"** untuk menjalankan prediksi — kamera
  akan berhenti otomatis setelah itu (tidak perlu klik tombol Stop bawaan
  komponen). Klik **"Rekam Ulang"** untuk mulai sesi baru.

## Langkah 2 — Jalankan Secara Lokal

```bash
# 1. Buat virtual environment (opsional tapi disarankan)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependency
pip install -r requirements.txt

# 3. Jalankan aplikasi
streamlit run app.py
```

Buka `http://localhost:8501` di browser. Untuk mode webcam, browser akan
meminta izin akses kamera — izinkan agar fitur ini berfungsi.

## Langkah 3 — Push ke GitHub

```bash
git init
git add .
git commit -m "Kamus SIBI - Prediksi Isyarat (Model Transformer NoAug)"
git branch -M main
git remote add origin https://github.com/<username>/<nama-repo>.git
git push -u origin main
```

> **Penting**: jika file model `.keras` berukuran lebih dari 100 MB, GitHub akan
> menolak push biasa. Gunakan [Git LFS](https://git-lfs.com/):
> ```bash
> git lfs install
> git lfs track "*.keras"
> git add .gitattributes
> ```

## Langkah 4 — Deploy ke Streamlit Community Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io) dan login dengan akun GitHub.
2. Klik **"New app"**.
3. Pilih repository, branch (`main`), dan file utama: `app.py`.
4. Klik **"Deploy"**. Streamlit Cloud otomatis membaca `requirements.txt` dan
   `packages.txt` untuk menginstall semua dependency.
5. Tunggu proses build selesai (biasanya 3–7 menit karena TensorFlow & MediaPipe
   cukup besar).
6. Aplikasi akan tersedia di URL seperti `https://<nama-app>.streamlit.app`.

## Mode Real-Time di Webcam

Tab Webcam sekarang punya toggle **"🔴 Prediksi otomatis (real-time)"** yang
aktif secara default — prediksi jalan sendiri setiap beberapa detik (bisa
diatur) tanpa perlu klik tombol. Cara kerjanya:

- Ekstraksi landmark MediaPipe dilakukan **sekali per frame saat frame itu
  masuk** dari kamera (bukan diulang setiap kali mau prediksi).
- Mode real-time memakai **subset kecil file bobot** (default 3 dari 15) yang
  digabung jadi satu graph model, supaya satu siklus prediksi tetap cepat.
- Kalau mau akurasi maksimal (ensemble penuh dari semua fold/seed), matikan
  toggle real-time dan pakai tombol "Prediksi dari Buffer Terakhir" — ini
  akan lebih lambat tapi memakai semua bobot yang dipilih.

## Kenapa Prediksi Bisa Terasa Lambat

Ada dua sumber utama:

1. **Jumlah file bobot yang di-ensemble.** Tiap file bobot = satu forward
   pass model. Makin banyak fold/seed yang dipakai sekaligus, makin lambat.
   Untuk kebutuhan cepat (real-time), pakai 2–5 file saja; untuk cek akurasi
   akhir, pakai semua.
2. **Re-ekstraksi MediaPipe berulang.** Di versi awal, seluruh buffer frame
   diproses ulang oleh MediaPipe setiap kali tombol prediksi ditekan — kerja
   dobel. Sekarang ekstraksi hanya dilakukan sekali per frame saat masuk,
   baik di mode Upload Video maupun Webcam.

Jika deploy di Streamlit Community Cloud (CPU saja, tanpa GPU), tetap
perkirakan ada beberapa ratus milidetik hingga 1–2 detik per siklus prediksi
tergantung berapa banyak bobot yang dipakai — ini batas wajar untuk inferensi
Transformer di CPU shared, bukan bug.

## Catatan tentang Mode Webcam

Mode webcam menggunakan `streamlit-webrtc`, yang memerlukan koneksi WebRTC
antara browser pengguna dan server. Ini berfungsi baik di sebagian besar
jaringan rumah, tetapi bisa gagal terhubung di jaringan kantor/kampus dengan
firewall ketat (karena hanya memakai STUN server publik, tanpa TURN server
berbayar). Jika webcam gagal terhubung, gunakan mode **Upload Video** sebagai
alternatif yang selalu berfungsi di semua jaringan.

## Batasan & Catatan Teknis

- Preprocessing (MediaPipe extraction → smoothing → normalisasi → fitur
  geometri) direplikasi persis dari notebook training agar distribusi input
  konsisten dengan data training.
- Prediksi dijalankan di CPU (Streamlit Cloud tidak menyediakan GPU gratis),
  sehingga sedikit lebih lambat dibanding saat training — umumnya tetap di
  bawah beberapa detik per klip pendek (45 frame setelah resampling).
- Daftar 17 kosakata mengikuti urutan alfabetis `LabelEncoder` saat training
  (lihat `utils/constants.py`). Jika urutan kelas asli kamu berbeda, sesuaikan
  `CLASS_NAMES` agar tetap cocok dengan output softmax model.
