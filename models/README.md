# Folder Model

Letakkan file **bobot** (bukan model utuh) hasil training Model 2
(Transformer, Tanpa Augmentasi) di folder `weights/`, contoh nama file:

```
models/weights/Model_2_iter1_seed42.weights.h5
models/weights/Model_2_iter2_seed42.weights.h5
models/weights/Model_2_iter3_seed123.weights.h5
models/weights/Model_2_iter4_seed123.weights.h5
models/weights/Model_2_iter5_seed777.weights.h5
... (boleh sebagian atau semua kombinasi fold/seed yang kamu punya)
```

Aplikasi akan otomatis mendeteksi SEMUA file `*.weights.h5` di folder ini,
membangun arsitektur Transformer untuk masing-masing, memuat bobotnya, lalu
mengambil rata-rata (ensemble) hasil prediksi softmax-nya saat inferensi.
Secara default, aplikasi hanya memilih file yang mengandung **"seed42"** di
namanya (5 fold, 1 seed) — lebih cepat dan tetap representatif untuk ke-5
fold. Kamu tetap bisa memilih subset lain dari UI aplikasi (menu Prediksi →
"Pengaturan model").

> Kenapa bukan `model.save()`? Karena file kamu disimpan dengan
> `model.save_weights(...)`, yang hanya menyimpan nilai bobot — arsitektur
> dibangun ulang di `utils/model_utils.py` (`build_transformer_model`) agar
> identik dengan saat training, baru bobotnya dimuat dengan `load_weights()`.
