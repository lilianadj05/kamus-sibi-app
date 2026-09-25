"""
Utilitas untuk memuat model Transformer (Model 2 - Tanpa Augmentasi) dan
menjalankan prediksi.

PENTING: bobot hasil training disimpan per fold-iterasi & per seed sebagai
file *.weights.h5 (bukan satu file model utuh), contoh:
    Model_2_iter4_seed123.weights.h5

File .weights.h5 HANYA berisi nilai bobot, bukan arsitektur. Karena itu,
arsitektur Transformer harus dibangun ulang dengan kode yang identik
dengan saat training, baru bobotnya dimuat dengan `load_weights()`.

Karena ada banyak kombinasi fold/seed, aplikasi ini memuat SEMUA file
*.weights.h5 yang ada di `WEIGHTS_DIR` lalu meng-ensemble prediksinya
(rata-rata softmax antar model) — hasil ini lebih stabil dibanding
memakai satu fold/seed saja, dan mendekati angka akurasi rata-rata
5-fold yang dilaporkan di notebook.
"""

import glob
import os

import numpy as np
import streamlit as st
import tensorflow as tf

from .constants import (
    CLASS_NAMES, WEIGHTS_DIR, MODEL_DROPOUT_RATE,
    TARGET_SEQ_LEN, FEATURE_DIM, NUM_CLASSES,
    OOD_CONFIDENCE_THRESHOLD, OOD_MARGIN_THRESHOLD, OOD_AGREEMENT_THRESHOLD,
)


# ════════════════════════════════════════════════════════════════
# CUSTOM LAYER (dipakai langsung sebagai objek Python saat build ulang
# arsitektur — tidak perlu deserialisasi, jadi cukup didefinisikan sama)
# ════════════════════════════════════════════════════════════════

class LearnedPositionalEncoding(tf.keras.layers.Layer):
    """Learned positional encoding via Embedding layer."""

    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        self.pos_emb = tf.keras.layers.Embedding(
            input_dim=max_len,
            output_dim=d_model,
            embeddings_initializer="glorot_uniform",
            name="pos_embedding",
        )

    def call(self, x):
        positions = tf.range(start=0, limit=self.max_len, delta=1)
        return x + self.pos_emb(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_len": self.max_len, "d_model": self.d_model})
        return cfg


# ════════════════════════════════════════════════════════════════
# ARSITEKTUR TRANSFORMER (identik dengan build_transformer_model di
# notebook — WAJIB sama persis agar bobot cocok saat load_weights)
# ════════════════════════════════════════════════════════════════

def build_transformer_model(seq_len=TARGET_SEQ_LEN,
                             feature_dim=FEATURE_DIM,
                             num_classes=NUM_CLASSES,
                             dropout_rate=MODEL_DROPOUT_RATE,
                             seed=42,
                             name="transformer_2block"):
    init = tf.keras.initializers.GlorotUniform(seed=seed)
    inputs = tf.keras.Input(shape=(seq_len, feature_dim), name=f"{name}_input")

    x = tf.keras.layers.Lambda(
        lambda t: t * tf.math.sqrt(tf.cast(feature_dim, tf.float32))
    )(inputs)
    x = LearnedPositionalEncoding(max_len=seq_len, d_model=feature_dim)(x)

    for _ in range(2):  # 2 attention block, sesuai notebook
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=2, key_dim=feature_dim // 2,
            dropout=dropout_rate, kernel_initializer=init,
        )(x, x)
        attn = tf.keras.layers.Dropout(dropout_rate)(attn)
        x = tf.keras.layers.Add()([x, attn])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        ffn = tf.keras.layers.Dense(feature_dim, activation="relu",
                                     kernel_initializer=init)(x)
        ffn = tf.keras.layers.Dropout(dropout_rate)(ffn)
        ffn = tf.keras.layers.Dense(feature_dim, activation=None,
                                     kernel_initializer=init)(ffn)
        x = tf.keras.layers.Add()([x, ffn])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(num_classes, activation="linear",
                               kernel_initializer=init)(x)
    outputs = tf.keras.layers.Activation("softmax", dtype="float32",
                                          name="softmax_output")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


# ════════════════════════════════════════════════════════════════
# DISCOVERY & LOADING BOBOT (di-cache — hanya dibangun sekali per sesi)
# ════════════════════════════════════════════════════════════════

def discover_weight_files(weights_dir: str = WEIGHTS_DIR):
    """Mencari semua file *.weights.h5 di dalam folder, terurut nama."""
    if not os.path.isdir(weights_dir):
        return []
    return sorted(glob.glob(os.path.join(weights_dir, "*.weights.h5")))


@st.cache_resource(show_spinner="Memuat model Transformer (ensemble)...")
def load_ensemble_models(weight_paths: tuple):
    """
    Membangun satu arsitektur Transformer per file bobot dan memuat
    bobotnya masing-masing. `weight_paths` berupa tuple (bukan list)
    agar bisa di-hash oleh st.cache_resource.

    Tiap submodel diberi nama unik (berdasarkan nama file bobotnya) —
    ini WAJIB agar beberapa submodel bisa digabung jadi satu graph di
    `load_combined_ensemble_model` tanpa bentrok nama.
    """
    models = []
    for i, path in enumerate(weight_paths):
        tag = os.path.splitext(os.path.splitext(os.path.basename(path))[0])[0]  # buang ".weights.h5"
        m = build_transformer_model(name=f"transformer_{tag}_{i}")
        m.load_weights(path)
        models.append(m)
    return models


@st.cache_resource(show_spinner="Menyiapkan model gabungan (ensemble)...")
def load_combined_ensemble_model(weight_paths: tuple):
    """
    Sama seperti `load_ensemble_models`, tapi seluruh sub-model digabung
    menjadi SATU graph Keras dengan DUA output:
      1. "average"      — rata-rata softmax antar submodel, (batch, NUM_CLASSES)
      2. "members"       — softmax tiap submodel TANPA dirata-rata,
                            (batch, n_model, NUM_CLASSES) — dipakai untuk
                            menghitung ensemble agreement (lihat
                            `predict_with_ood_check`).
    Dengan begini, ensemble + diagnostik out-of-vocabulary tetap hanya
    butuh SATU panggilan `predict()`, bukan N panggilan terpisah.
    """
    sub_models = load_ensemble_models(weight_paths)
    shared_input = tf.keras.Input(shape=(TARGET_SEQ_LEN, FEATURE_DIM), name="input")
    member_outputs = [m(shared_input) for m in sub_models]  # tiap: (batch, NUM_CLASSES)

    if len(member_outputs) == 1:
        averaged = member_outputs[0]
    else:
        averaged = tf.keras.layers.Average(name="ensemble_average")(member_outputs)

    stacked = tf.keras.layers.Lambda(
        lambda ts: tf.stack(ts, axis=1), name="ensemble_members"
    )(member_outputs)  # (batch, n_model, NUM_CLASSES)

    return tf.keras.Model(
        inputs=shared_input, outputs=[averaged, stacked], name="ensemble_combined"
    )


def predict_single(model, model_input: np.ndarray, top_k: int = 5):
    """Prediksi dengan SATU model (kompatibilitas lama — model dengan
    satu output softmax, bukan hasil `load_combined_ensemble_model`)."""
    probs = model.predict(model_input, verbose=0)[0]
    order = np.argsort(probs)[::-1][:top_k]
    return [(CLASS_NAMES[i], float(probs[i])) for i in order]


def predict_with_ood_check(combined_model, model_input: np.ndarray, top_k: int = 5):
    """
    Menjalankan prediksi ensemble + diagnostik out-of-vocabulary (OOV).
    `combined_model` HARUS berasal dari `load_combined_ensemble_model`
    (dua output: averaged & members).

    Returns:
        results: daftar top_k (label, probabilitas) dari rata-rata ensemble
        diagnostics: dict berisi confidence, margin, agreement, dan
                     is_confident (bool — False berarti sebaiknya
                     ditampilkan sebagai "Isyarat Tidak Dikenali")
    """
    averaged, members = combined_model.predict(model_input, verbose=0)
    averaged = averaged[0]        # (NUM_CLASSES,)
    members = members[0]          # (n_model, NUM_CLASSES)

    order = np.argsort(averaged)[::-1]
    top1_idx = int(order[0])
    top1_prob = float(averaged[top1_idx])
    top2_prob = float(averaged[order[1]]) if len(order) > 1 else 0.0
    margin = top1_prob - top2_prob

    member_top1 = np.argmax(members, axis=-1)  # (n_model,)
    agreement = float(np.mean(member_top1 == top1_idx)) if len(member_top1) else 0.0

    is_confident = (
        top1_prob >= OOD_CONFIDENCE_THRESHOLD
        and margin >= OOD_MARGIN_THRESHOLD
        and agreement >= OOD_AGREEMENT_THRESHOLD
    )

    results = [(CLASS_NAMES[i], float(averaged[i])) for i in order[:top_k]]
    diagnostics = {
        "confidence": top1_prob,
        "margin": margin,
        "agreement": agreement,
        "n_models": int(members.shape[0]),
        "is_confident": is_confident,
    }
    return results, diagnostics


def predict(models, model_input: np.ndarray, top_k: int = 5):
    """
    Menjalankan prediksi pada semua model dalam ensemble (list terpisah),
    merata-ratakan probabilitas softmax-nya, lalu mengembalikan top_k
    (label, probabilitas) terurut menurun. Lebih lambat dibanding
    `predict_single` + `load_combined_ensemble_model` karena N panggilan
    predict() terpisah — dipertahankan untuk mode Upload Video di mana
    latensi kurang krusial.
    """
    all_probs = np.stack([
        m.predict(model_input, verbose=0)[0] for m in models
    ], axis=0)  # (n_model, NUM_CLASSES)
    probs = all_probs.mean(axis=0)

    order = np.argsort(probs)[::-1][:top_k]
    return [(CLASS_NAMES[i], float(probs[i])) for i in order]
