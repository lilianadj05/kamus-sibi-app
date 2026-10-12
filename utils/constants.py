"""
Konstanta global untuk aplikasi Prediksi Isyarat SIBI.
Nilai-nilai ini HARUS sama persis dengan yang dipakai saat training.

Model yang dipakai saat ini: MODEL 4 — Transformer DENGAN Augmentasi
(lihat notebook: Copy_of_R2_HANDS3D_Geometry164_seq45_AugTransformer_3seed).
Augmentasi hanya berlaku saat training (memperbanyak data), TIDAK
memengaruhi pipeline preprocessing saat inferensi di app ini.
"""

# ── Parameter fitur & sekuens (harus sama dengan training) ─────────
TARGET_SEQ_LEN  = 45     # jumlah frame setelah resampling
FEATURE_DIM     = 164    # 126 XYZ + 19 geometri tangan kanan + 19 geometri tangan kiri
RAW_FEATURE_DIM = 126    # 21 landmark x 3 (XYZ) x 2 tangan
NUM_CLASSES     = 17

RHAND_SLICE = slice(0, 63)     # tangan kanan: 21 landmark x 3
LHAND_SLICE = slice(63, 126)   # tangan kiri : 21 landmark x 3

# ── Konfigurasi MediaPipe Hands (sama dengan training) ──────────────
MEDIAPIPE_CONFIG = dict(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# ── Daftar kelas (URUTAN HARUS SAMA dengan le.classes_ saat training) ──
# LabelEncoder mengurutkan kelas secara alfabetis, jadi urutan di
# bawah ini mengikuti urutan alfabetis dari 17 kosakata pada dataset.
CLASS_NAMES = [
    "AIDS", "Buah Zakar", "Dubur", "Ereksi", "Gairah", "Gonorrhoeae",
    "HIV", "Hamil", "Hormon", "IMS", "IUD", "Impoten", "Penis",
    "Rahim", "Seks", "Selaput Dara", "Vagina",
]

# Kosakata yang dikecualikan dari evaluasi 14-kelas pada eksperimen
# (tetap disertakan dalam prediksi karena aplikasi ini untuk edukasi kesehatan)
KATA_SENSITIF = ["Buah Zakar", "Penis", "Ereksi"]

# ── Info ringkas model (hasil evaluasi akhir dari notebook, 15 run:
# 5-fold x 3 seed, kelas 17, dihitung dari output training Model 4) ──
MODEL_INFO = {
    "nama": "Transformer (Dengan Augmentasi) + Fitur Geometri 164-D",
    "arsitektur": "Transformer Encoder 2-Block",
    "augmentasi": True,
    "avg_accuracy": 0.8559,
    "avg_f1": 0.8543,
    "std_accuracy": 0.0183,
    "seq_len": TARGET_SEQ_LEN,
    "feature_dim": FEATURE_DIM,
    "jumlah_kelas": NUM_CLASSES,
}

# ── Konfigurasi bobot model (disimpan sebagai .weights.h5 per ──────
# fold-iterasi & per seed, bukan sebagai satu file model utuh). Hanya
# file dengan prefix "Model_4_" yang dimuat & di-ensemble (rata-rata
# softmax) saat prediksi — supaya file bobot Model 2 lama yang
# mungkin masih nyangkut di folder ini tidak ikut terpakai.
WEIGHTS_DIR = "models/weights"
WEIGHT_FILE_PATTERN = "Model_4_*.weights.h5"

# Seed default yang dipakai (5-fold, seed ini saja) kalau kamu ingin
# menyaring manual — 1 seed penuh 5-fold jauh lebih cepat daripada
# 3 seed x 5 fold (15 file), dengan penurunan akurasi yang minim karena
# variasi antar-seed jauh lebih kecil dibanding variasi antar-fold.
DEFAULT_SEED_TAG = "seed42"

# Hyperparameter arsitektur Model 4 (HARUS sama persis dengan saat
# training agar bobot bisa dimuat) — lihat MODEL_VARIANTS di notebook:
# {"id":"Model_4", "arsitektur":"transformer", "augmentasi":True, "dropout":0.4}
# PENTING: Model 4 dilatih dengan arsitektur yang sudah diperbaiki —
# TANPA scaling input oleh √feature_dim sebelum positional encoding
# (scaling itu adalah bug di Model 2/versi lama, sudah dihapus di sini).
MODEL_DROPOUT_RATE = 0.4

# ── Konfigurasi buffer webcam (fixed, tidak bisa diubah dari UI) ────
WEBCAM_BUFFER_SECONDS = 6
WEBCAM_APPROX_FPS = 15

# Rasio minimum frame dengan tangan terdeteksi di buffer agar prediksi
# dijalankan. Di bawah ini, hasil langsung "Tidak Terdeteksi" tanpa
# menjalankan model (mencegah hasil "default" dari input kosong/nol).
MIN_HAND_DETECTION_RATIO = 0.05

# ── Ambang penolakan untuk input di luar 17 kosakata (out-of-vocabulary) ──
# Model bersifat closed-set: softmax SELALU memaksa memilih salah satu
# dari 17 kelas, walau gerakannya di luar kosakata itu. Kombinasi 3 sinyal
# di bawah dipakai supaya penolakan lebih tepercaya dibanding threshold
# confidence tunggal:
#   1. confidence  — probabilitas softmax top-1 (rata-rata ensemble)
#   2. margin      — selisih probabilitas top-1 vs top-2 (makin tipis,
#                     makin ragu modelnya)
#   3. agreement   — proporsi submodel (fold) yang argmax-nya SAMA dengan
#                     kelas top-1 hasil rata-rata (makin rendah, makin
#                     besar kemungkinan fold-fold saling tidak sepakat —
#                     tanda kuat input di luar distribusi training)
# Ketiganya harus lolos ambang agar prediksi dianggap "yakin"; kalau salah
# satu gagal, hasil ditampilkan sebagai "Isyarat Tidak Dikenali".
OOD_CONFIDENCE_THRESHOLD = 0.35
OOD_MARGIN_THRESHOLD = 0.12
OOD_AGREEMENT_THRESHOLD = 0.6
