"""
Pipeline preprocessing untuk inferensi — direplikasi persis dari notebook
training (HANDS3D_Geometry164_seq45_NoAugTransforme_3seed) agar model
menerima input dengan distribusi yang sama seperti saat dilatih.

Urutan pipeline untuk satu klip (video upload / rekaman webcam):
  1. Ekstraksi landmark MediaPipe per frame  -> (T, 126)
  2. Trim frame idle di awal/akhir klip
  3. Smoothing channel Z (moving average, window=3)
  4. Resampling temporal ke 45 frame (nearest-neighbor)
  5. Normalisasi spasial (wrist-relative + scale) per tangan
  6. Pengayaan fitur geometri (19 fitur/tangan)         -> (45, 164)
"""

import math
import numpy as np
import cv2
import mediapipe as mp

from .constants import (
    TARGET_SEQ_LEN, FEATURE_DIM, RAW_FEATURE_DIM,
    RHAND_SLICE, LHAND_SLICE, MEDIAPIPE_CONFIG,
)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


# ════════════════════════════════════════════════════════════════
# 1. EKSTRAKSI LANDMARK MEDIAPIPE
# ════════════════════════════════════════════════════════════════

def _safe_xyz(landmark_list, idx):
    if landmark_list is None:
        return 0.0, 0.0, 0.0
    try:
        lm = landmark_list.landmark[idx]
        return float(lm.x), float(lm.y), float(lm.z)
    except Exception:
        return 0.0, 0.0, 0.0


def extract_hand_features_xyz(hand_landmarks):
    """21 landmark x (x, y, z) = 63 nilai untuk satu tangan."""
    if hand_landmarks is None:
        return [0.0] * 63
    return [c for i in range(21) for c in _safe_xyz(hand_landmarks, i)]


def extract_frame_features_xyz(results):
    """
    Mengembalikan vektor 126-dim per frame: [tangan_kanan(63)] + [tangan_kiri(63)].
    Tangan yang tidak terdeteksi -> blok nol.
    """
    r_hand, l_hand = [0.0] * 63, [0.0] * 63
    if results.multi_hand_landmarks:
        for idx, handedness in enumerate(results.multi_handedness):
            lms = results.multi_hand_landmarks[idx]
            label = handedness.classification[0].label
            if label == "Right":
                r_hand = extract_hand_features_xyz(lms)
            else:
                l_hand = extract_hand_features_xyz(lms)
    return np.array(r_hand + l_hand, dtype=np.float32)


def process_frames_xyz(frames_bgr, draw_last_frame=False):
    """
    Menjalankan MediaPipe Hands pada daftar frame BGR (numpy array).
    Instance MediaPipe baru dibuat khusus untuk klip ini (mencegah state
    leakage antar klip, sama seperti pipeline training).

    Returns:
        raw_seq: (T, 126) float32
        detection_ratio: proporsi frame dengan minimal satu tangan terdeteksi
        annotated_last: frame terakhir dengan landmark digambar (jika diminta)
    """
    raw_frames = []
    detected_count = 0
    annotated_last = None

    with mp_hands.Hands(**MEDIAPIPE_CONFIG) as hands_model:
        for i, frame in enumerate(frames_bgr):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_model.process(rgb)
            feat = extract_frame_features_xyz(results)
            raw_frames.append(feat)
            if results.multi_hand_landmarks:
                detected_count += 1
            if draw_last_frame and i == len(frames_bgr) - 1:
                annotated = frame.copy()
                if results.multi_hand_landmarks:
                    for hlm in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated, hlm, mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style(),
                        )
                annotated_last = annotated

    raw_seq = np.array(raw_frames, dtype=np.float32) if raw_frames else \
        np.empty((0, RAW_FEATURE_DIM), dtype=np.float32)
    detection_ratio = detected_count / len(frames_bgr) if frames_bgr else 0.0
    return raw_seq, detection_ratio, annotated_last


# ════════════════════════════════════════════════════════════════
# 2. TRIM FRAME IDLE (motion energy)
# ════════════════════════════════════════════════════════════════

def compute_motion_energy(seq, part_slices=(RHAND_SLICE, LHAND_SLICE)):
    if seq.shape[0] < 2:
        return np.zeros(seq.shape[0], dtype=np.float32)
    energy = np.zeros(seq.shape[0], dtype=np.float32)
    for sl in part_slices:
        block = seq[:, sl]
        diffs = np.diff(block, axis=0)
        mag = np.linalg.norm(diffs.reshape(diffs.shape[0], -1, 3), axis=2)
        energy[1:] += mag.mean(axis=1)
    return energy


def trim_idle_frames(seq, threshold_ratio=0.15, smooth_window=5,
                      padding=3, min_frames=TARGET_SEQ_LEN):
    T = seq.shape[0]
    if T <= min_frames:
        return seq
    energy = compute_motion_energy(seq)
    energy_smooth = np.convolve(
        energy, np.ones(smooth_window) / smooth_window, mode="same"
    ) if smooth_window > 1 else energy

    max_e = energy_smooth.max()
    if max_e <= 1e-8:
        return seq

    active_idx = np.where(energy_smooth > threshold_ratio * max_e)[0]
    if len(active_idx) == 0:
        return seq

    start = max(0, int(active_idx[0]) - padding)
    end = min(T, int(active_idx[-1]) + padding + 1)
    if (end - start) < min_frames:
        return seq
    return seq[start:end]


# ════════════════════════════════════════════════════════════════
# 3. SMOOTHING CHANNEL Z
# ════════════════════════════════════════════════════════════════

def smooth_z_channel(seq, window_size=3):
    if seq.shape[0] < window_size:
        return seq
    smoothed = seq.copy()
    kernel = np.ones(window_size) / window_size
    for z_idx in range(2, seq.shape[1], 3):
        smoothed[:, z_idx] = np.convolve(seq[:, z_idx], kernel, mode="same")
    return smoothed


# ════════════════════════════════════════════════════════════════
# 4. RESAMPLING TEMPORAL
# ════════════════════════════════════════════════════════════════

def temporal_normalize(seq, target_len=TARGET_SEQ_LEN):
    T = seq.shape[0]
    if T == 0:
        return np.zeros((target_len, seq.shape[1]), dtype=np.float32)
    indices = [min(int(math.floor(i * T / target_len)), T - 1)
               for i in range(target_len)]
    return seq[indices].astype(np.float32)


# ════════════════════════════════════════════════════════════════
# 5. NORMALISASI SPASIAL (wrist-relative + scale)
# ════════════════════════════════════════════════════════════════

def spatial_normalize_hands(seq):
    if seq.shape[0] == 0:
        return seq
    norm_seq = np.zeros_like(seq)

    for hand_offset in [0, 63]:
        hand_slice = slice(hand_offset, hand_offset + 63)
        hand_data = seq[:, hand_slice]

        if not np.any(hand_data != 0.0):
            continue

        wrist_coords = hand_data[:, 0:3]
        centered = np.zeros_like(hand_data)
        for lm_i in range(0, 63, 3):
            centered[:, lm_i:lm_i + 3] = hand_data[:, lm_i:lm_i + 3] - wrist_coords

        finger_coords = centered[:, 3:].reshape(-1, 3)
        non_zero_mask = np.any(finger_coords != 0.0, axis=1)
        non_zero = finger_coords[non_zero_mask]

        if len(non_zero) == 0:
            norm_seq[:, hand_slice] = centered
            continue

        max_dist = np.max(np.linalg.norm(non_zero, axis=1))
        if max_dist < 1e-6:
            norm_seq[:, hand_slice] = centered
        else:
            norm_seq[:, hand_slice] = centered / max_dist

    return norm_seq.astype(np.float32)


# ════════════════════════════════════════════════════════════════
# 6. PENGAYAAN FITUR GEOMETRI (19 fitur/tangan)
# ════════════════════════════════════════════════════════════════

def compute_hand_geometry(hand_xyz):
    if not np.any(hand_xyz != 0):
        return np.zeros(19, dtype=np.float32)

    wrist = hand_xyz[0]
    fingertips = [4, 8, 12, 16, 20]
    mcp_joints = [1, 5, 9, 13, 17]
    features = []

    for tip in fingertips:
        features.append(np.linalg.norm(hand_xyz[tip] - wrist))

    for i in range(len(fingertips) - 1):
        features.append(np.linalg.norm(hand_xyz[fingertips[i]] - hand_xyz[fingertips[i + 1]]))

    for tip, mcp in zip(fingertips, mcp_joints):
        tip_mcp = np.linalg.norm(hand_xyz[tip] - hand_xyz[mcp])
        mcp_wrist = np.linalg.norm(hand_xyz[mcp] - wrist) + 1e-6
        features.append(tip_mcp / mcp_wrist)

    v1 = hand_xyz[5] - hand_xyz[0]
    v2 = hand_xyz[17] - hand_xyz[0]
    normal = np.cross(v1, v2)
    mag = np.linalg.norm(normal) + 1e-6
    features.extend((normal / mag).tolist())

    features.append(np.linalg.norm(hand_xyz[4] - hand_xyz[8]))

    palm_center = hand_xyz[[0, 5, 9, 13, 17]].mean(axis=0)
    features.append(np.mean([np.linalg.norm(hand_xyz[t] - palm_center) for t in fingertips]))

    return np.array(features, dtype=np.float32)


def enrich_sequence_with_geometry(seq_126):
    T = seq_126.shape[0]
    enriched = np.zeros((T, FEATURE_DIM), dtype=np.float32)
    enriched[:, :126] = seq_126
    for t in range(T):
        r_xyz = seq_126[t, 0:63].reshape(21, 3)
        l_xyz = seq_126[t, 63:126].reshape(21, 3)
        enriched[t, 126:145] = compute_hand_geometry(r_xyz)
        enriched[t, 145:164] = compute_hand_geometry(l_xyz)
    return enriched


def make_live_hands_model():
    """Membuat satu instance MediaPipe Hands untuk dipakai berulang kali
    selama stream webcam berlangsung (kebalikan dari process_frames_xyz
    yang membuat instance baru per klip — di sini kita justru INGIN satu
    instance persisten sepanjang stream, supaya tracking antar-frame
    MediaPipe bekerja optimal & tidak perlu re-init tiap frame)."""
    return mp_hands.Hands(**MEDIAPIPE_CONFIG)


def extract_and_annotate(frame_bgr, hands_model, draw=True):
    """
    Menjalankan MediaPipe pada SATU frame (dipanggil dari callback webrtc
    setiap frame masuk). Mengembalikan fitur 126-dim, apakah tangan
    terdeteksi, dan frame dengan overlay status deteksi (+ bounding box
    & landmark jika tangan terdeteksi) supaya pengguna dapat feedback
    visual langsung saat merekam.
    """
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = hands_model.process(rgb)
    feat = extract_frame_features_xyz(results)
    detected = bool(results.multi_hand_landmarks)

    annotated = frame_bgr.copy() if draw else frame_bgr

    if draw and detected:
        for hlm in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated, hlm, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )
            # ── Bounding box di sekitar tangan yang terdeteksi ──────
            xs = [lm.x for lm in hlm.landmark]
            ys = [lm.y for lm in hlm.landmark]
            pad = 0.04
            x1 = max(0, int((min(xs) - pad) * w))
            y1 = max(0, int((min(ys) - pad) * h))
            x2 = min(w, int((max(xs) + pad) * w))
            y2 = min(h, int((max(ys) + pad) * h))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)

    if draw:
        # ── Label status deteksi tangan (pojok kiri atas) ───────────
        if detected:
            label, color = "Tangan Terdeteksi", (0, 170, 0)
        else:
            label, color = "Tangan Tidak Terdeteksi", (0, 0, 220)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (8, 8), (18 + tw, 40 + th), (255, 255, 255), -1)
        cv2.putText(annotated, label, (14, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    return feat, detected, annotated


# ════════════════════════════════════════════════════════════════
# PIPELINE LENGKAP — dipanggil dari app.py
# ════════════════════════════════════════════════════════════════

def raw_seq_to_model_input(raw_seq):
    """
    Menjalankan tahap 2-6 pipeline (trim -> smooth -> resample -> normalize
    -> geometry) pada array fitur mentah (T, 126) yang SUDAH diekstrak
    sebelumnya oleh MediaPipe. Dipisah dari ekstraksi MediaPipe supaya bisa
    dipakai ulang untuk mode live (fitur sudah diekstrak per frame saat
    frame itu masuk, tidak perlu diekstrak ulang tiap kali mau prediksi).

    Returns: (1, 45, 164) float32, atau None jika raw_seq kosong
    """
    if raw_seq is None or raw_seq.shape[0] == 0:
        return None
    trimmed = trim_idle_frames(raw_seq)
    z_smoothed = smooth_z_channel(trimmed)
    resampled = temporal_normalize(z_smoothed, TARGET_SEQ_LEN)
    normalized = spatial_normalize_hands(resampled)
    enriched = enrich_sequence_with_geometry(normalized)
    return enriched[np.newaxis, ...].astype(np.float32)  # (1, 45, 164)


def preprocess_frames(frames_bgr, draw_last_frame=False):
    """
    Menjalankan seluruh pipeline preprocessing pada daftar frame BGR
    (dipakai untuk mode Upload Video, di mana ekstraksi MediaPipe belum
    pernah dilakukan sebelumnya). Untuk mode webcam live, gunakan
    `raw_seq_to_model_input` langsung pada buffer fitur yang sudah
    diekstrak di callback webrtc.

    Returns:
        model_input: (1, 45, 164) float32, atau None jika tidak ada frame
        info: dict berisi detection_ratio, n_frame_asli, annotated_frame
    """
    if not frames_bgr:
        return None, {"detection_ratio": 0.0, "n_frame_asli": 0, "annotated_frame": None}

    raw_seq, detection_ratio, annotated = process_frames_xyz(
        frames_bgr, draw_last_frame=draw_last_frame
    )

    if raw_seq.shape[0] == 0:
        return None, {"detection_ratio": 0.0, "n_frame_asli": 0, "annotated_frame": annotated}

    model_input = raw_seq_to_model_input(raw_seq)
    info = {
        "detection_ratio": detection_ratio,
        "n_frame_asli": raw_seq.shape[0],
        "annotated_frame": annotated,
    }
    return model_input, info
