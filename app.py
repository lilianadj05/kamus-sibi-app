"""
Kamus SIBI — Prediksi Isyarat (Model Transformer, Dengan Augmentasi)
Aplikasi Streamlit untuk mengenali kosakata Bahasa Isyarat Indonesia (SIBI)
dari video unggahan maupun webcam, menggunakan model Transformer 2-Block
(Model 4) yang dilatih dengan augmentasi geometry-consistent pada fitur
geometri tangan 164-dimensi (MediaPipe Hands).
"""

import collections
import os
import tempfile
import threading

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from utils.constants import (
    CLASS_NAMES, MODEL_INFO, WEIGHTS_DIR, KATA_SENSITIF,
    WEBCAM_BUFFER_SECONDS, WEBCAM_APPROX_FPS, MIN_HAND_DETECTION_RATIO,
)
from utils.preprocessing import (
    preprocess_frames, raw_seq_to_model_input,
    make_live_hands_model, extract_and_annotate,
)
from utils.model_utils import discover_weight_files, load_combined_ensemble_model, predict_with_ood_check
from utils.ui import inject_global_css, render_topbar, section_header, ACCENT_BLUE

# UNTUK CLOUD TAMBAH INIIIIIIIII!!!!!!
from twilio.rest import Client

@st.cache_data(ttl=3000)
def get_ice_servers():
    """Generate TURN credentials dari Twilio."""
    try:
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        api_key = st.secrets["TWILIO_API_KEY"]
        api_secret = st.secrets["TWILIO_API_SECRET"]
    except KeyError as e:
        st.error(f"Kredensial Twilio belum di-set: {e}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]
    
    client = Client(api_key, api_secret, account_sid)
    token = client.tokens.create(ttl=3600)
    
    ice_servers = []
    for server in token.ice_servers:
        entry = {"urls": server["urls"]}
        if "username" in server:
            entry["username"] = server["username"]
            entry["credential"] = server["credential"]
        ice_servers.append(entry)
    
    return ice_servers

st.set_page_config(
    page_title="Kamus SIBI - Prediksi Isyarat",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_css()


# ════════════════════════════════════════════════════════════════
# HELPER RENDER HASIL PREDIKSI (dipakai Upload Video & Webcam)
# ════════════════════════════════════════════════════════════════

def render_badge_card(badge_label, confidence=None, extra_caption=""):
    conf_html = (
        f'<p style="margin-top:10px;">Keyakinan: <b>{confidence*100:.1f}%</b></p>'
        if confidence is not None else ""
    )
    st.markdown(f"""
        <div class="sibi-card" style="text-align:center;">
        <p class="sibi-caption">Hasil Prediksi</p>
        <span class="sibi-result-badge">{badge_label}</span>
        {conf_html}
        <p class="sibi-caption">{extra_caption}</p>
        </div>
    """, unsafe_allow_html=True)


def render_diagnostics_caption(diagnostics):
    n_agree = int(round(diagnostics["agreement"] * diagnostics["n_models"]))
    st.caption(
        f"Diagnostik ensemble — confidence: {diagnostics['confidence']*100:.1f}% · "
        f"margin top-1/top-2: {diagnostics['margin']*100:.1f}% · "
        f"kesepakatan fold: {n_agree}/{diagnostics['n_models']}"
    )


def render_topk_list(results):
    section_header("Top-5 Kemungkinan Kosakata")
    for label, prob in results:
        st.write(f"**{label}**")
        st.progress(min(prob, 1.0), text=f"{prob*100:.1f}%")


def maybe_show_sensitive_note(label):
    if label in KATA_SENSITIF:
        st.info(
            "ℹ️ Kosakata ini termasuk istilah kesehatan reproduksi yang "
            "digunakan untuk konteks edukasi kesehatan bagi komunitas Tuli."
        )

# ════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGASI
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sibi-sidebar-title">🤟 Kamus SIBI</div>', unsafe_allow_html=True)
    page = option_menu(
        menu_title=None,
        options=["Home", "Petunjuk", "Prediksi", "Tentang Model"],
        icons=["house", "info-circle", "camera-video", "bar-chart-line"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "#2b3542"},
            "icon": {"color": "#c7ccd4", "font-size": "16px"},
            "nav-link": {
                "font-size": "15px", "color": "#e4e7eb", "text-align": "left",
                "margin": "2px 0", "padding": "10px 16px", "border-radius": "0px",
                "--hover-color": "#232b36",
            },
            "nav-link-selected": {"background-color": "#1e88e5", "color": "white", "font-weight": "600"},
        },
    )

# ════════════════════════════════════════════════════════════════
# HALAMAN: HOME
# ════════════════════════════════════════════════════════════════
if page == "Home":
    render_topbar("Home", "Beranda")
    st.markdown(f"""
        <div class="sibi-card">
        <h3 style="margin-top:0;">Selamat datang di Kamus SIBI — Prediksi Isyarat</h3>
        <p>Aplikasi ini mengenali kosakata Bahasa Isyarat Indonesia (SIBI) dari
        gerakan tangan menggunakan model <b>Transformer Encoder</b> yang dilatih pada
        fitur geometri tangan 164-dimensi hasil ekstraksi MediaPipe Hands.
        Aplikasi mendukung dua mode input: unggah video dan webcam langsung.</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Akurasi Model (rata-rata 5-fold)", f"{MODEL_INFO['avg_accuracy']*100:.1f}%")
    col2.metric("F1-Score Macro", f"{MODEL_INFO['avg_f1']*100:.1f}%")
    col3.metric("Jumlah Kosakata", MODEL_INFO["jumlah_kelas"])

    section_header("Daftar Kosakata yang Dikenali")
    cols = st.columns(4)
    for i, name in enumerate(CLASS_NAMES):
        cols[i % 4].markdown(f"- {name}")

# ════════════════════════════════════════════════════════════════
# HALAMAN: PETUNJUK
# ════════════════════════════════════════════════════════════════
elif page == "Petunjuk":
    render_topbar("Petunjuk Penggunaan", "Home / Petunjuk")
    section_header("Cara Menggunakan Aplikasi")
    st.markdown("""
        <div class="sibi-card">
        <ol>
        <li>Buka menu <b>Prediksi</b> di sidebar.</li>
        <li>Pilih tab <b>Upload Video</b> atau <b>Webcam</b>.</li>
        <li>Untuk upload video: pilih file berformat MP4/MOV/AVI berisi satu gerakan isyarat.</li>
        <li>Untuk webcam: klik <b>START</b> di bawah video, izinkan akses kamera, lakukan
            gerakan isyarat, lalu klik <b>STOP</b> — prediksi otomatis muncul.</li>
        <li>Sistem akan menampilkan kosakata hasil prediksi beserta tingkat keyakinan (confidence).</li>
        </ol>
        </div>
        <div class="sibi-card">
        <h4 style="margin-top:0;">Tips agar prediksi lebih akurat</h4>
        <ul>
        <li>Pastikan pencahayaan cukup terang dan latar belakang tidak terlalu ramai.</li>
        <li>Posisikan tangan sepenuhnya terlihat dalam bingkai kamera.</li>
        <li>Lakukan gerakan isyarat dengan durasi sekitar 2–4 detik, tidak terlalu cepat.</li>
        <li>Jika hasil deteksi tangan rendah (lihat indikator "Rasio Deteksi Tangan"),
            coba ulangi rekaman dengan posisi tangan lebih jelas.</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# HALAMAN: PREDIKSI
# ════════════════════════════════════════════════════════════════
elif page == "Prediksi":
    render_topbar("Prediksi Isyarat", "Home / Prediksi")

    all_weight_files = discover_weight_files(WEIGHTS_DIR)
    if not all_weight_files:
        st.error(
            f"⚠️ Tidak ada file bobot (Model_4_*.weights.h5) di folder `{WEIGHTS_DIR}`. "
            "Letakkan file seperti `Model_4_iter4_seed123.weights.h5` hasil "
            "training Model 4 (Transformer, Dengan Augmentasi) di folder tersebut."
        )
        st.stop()

    st.caption(f"Model dimuat dari {len(all_weight_files)} file bobot di `{WEIGHTS_DIR}` (semua di-ensemble).")

    combined_model = load_combined_ensemble_model(tuple(all_weight_files))

    st.session_state.setdefault("input_mode", "📁 Upload Video")
    input_mode = st.radio(
        "Mode input", ["📁 Upload Video", "📷 Webcam"],
        horizontal=True, label_visibility="collapsed", key="input_mode",
    )

    # ── MODE: UPLOAD VIDEO ───────────────────────────────────────
    if input_mode == "📁 Upload Video":
        section_header("Unggah Video Isyarat")
        video_file = st.file_uploader(
            "Pilih file video (MP4, MOV, AVI) berisi satu gerakan isyarat",
            type=["mp4", "mov", "avi"],
        )

        if video_file is not None:
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.video(video_file)

            if st.button("🔍 Prediksi Video Ini", type="primary"):
                with st.spinner("Mengekstrak landmark tangan & menjalankan prediksi..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                        tmp.write(video_file.read())
                        tmp_path = tmp.name

                    cap = cv2.VideoCapture(tmp_path)
                    frames = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames.append(frame)
                    cap.release()

                    model_input, info = preprocess_frames(frames, draw_last_frame=True)

                    if model_input is None:
                        st.warning("Tidak ada frame yang berhasil dibaca dari video ini.")
                    else:
                        extra_caption = (
                            f"Rasio deteksi tangan: {info['detection_ratio']*100:.0f}% "
                            f"| Frame asli: {info['n_frame_asli']}"
                        )

                        if info["detection_ratio"] < MIN_HAND_DETECTION_RATIO:
                            with col_b:
                                render_badge_card("Tidak Terdeteksi", extra_caption=extra_caption)
                            st.warning(
                                "Tangan tidak terdeteksi di video ini. Coba unggah video "
                                "dengan tangan lebih terlihat jelas dan pencahayaan lebih baik."
                            )
                        else:
                            results, diag = predict_with_ood_check(combined_model, model_input, top_k=5)
                            is_ok = diag["is_confident"]
                            badge_label = results[0][0] if is_ok else "Isyarat Tidak Dikenali"

                            with col_b:
                                render_badge_card(badge_label, confidence=diag["confidence"], extra_caption=extra_caption)
                            render_diagnostics_caption(diag)

                            if not is_ok:
                                st.warning(
                                    "Gerakan ini tidak cukup meyakinkan cocok dengan salah satu "
                                    "dari 17 kosakata yang dikenali. Berikut kemungkinan terdekat "
                                    "(belum tentu benar):"
                                )

                            render_topk_list(results)

                            if is_ok:
                                maybe_show_sensitive_note(results[0][0])

    # ── MODE: WEBCAM ─────────────────────────────────────────────
    elif input_mode == "📷 Webcam":
        section_header("Prediksi via Webcam")
        st.markdown(
            '<p class="sibi-caption">Klik <b>START</b> di bawah video, izinkan akses kamera, lalu '
            'lakukan gerakan isyarat (kotak hijau & tulisan di video menandakan tangan sudah '
            'terdeteksi). Klik <b>STOP</b> saat selesai — prediksi otomatis muncul begitu kamera '
            'berhenti.</p>',
            unsafe_allow_html=True,
        )

        buffer_len = WEBCAM_BUFFER_SECONDS * WEBCAM_APPROX_FPS  # fixed, tidak ada opsi

        st.session_state.setdefault("cam_was_playing", False)
        st.session_state.setdefault("cam_processor_ref", None)
        st.session_state.setdefault("webcam_result", None)

        # ── Video processor: ekstrak landmark, status deteksi per frame ────
        class HandBufferProcessor:
            def __init__(self):
                self.lock = threading.Lock()
                self.feature_buffer = collections.deque(maxlen=buffer_len)
                self.detected_buffer = collections.deque(maxlen=buffer_len)
                self.last_annotated = None
                self._hands_model = make_live_hands_model()

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                feat, detected, annotated = extract_and_annotate(
                    img, self._hands_model, draw=True
                )
                with self.lock:
                    self.feature_buffer.append(feat)
                    self.detected_buffer.append(detected)
                    self.last_annotated = annotated
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        # UNTUK CLOUD TAMBAH INIIIIIIIII!!!!!!
        ctx = webrtc_streamer(
            key="sibi-webcam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": get_ice_servers()},  # ← TAMBAHKAN BARIS INI
            video_processor_factory=HandBufferProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        was_playing = st.session_state.cam_was_playing
        now_playing = ctx.state.playing

        # Sesi baru dimulai (user klik START) -> bersihkan hasil lama
        if now_playing and not was_playing:
            st.session_state.webcam_result = None

        # Simpan referensi processor SELAGI kamera masih aktif. WAJIB:
        # begitu STOP diklik, library ini langsung melepas ctx.video_processor
        # (di dalam pemanggilan webrtc_streamer() itu sendiri) sebelum kode
        # kita sempat membaca buffer-nya — jadi harus disimpan lebih dulu.
        if now_playing and ctx.video_processor is not None:
            st.session_state.cam_processor_ref = ctx.video_processor

        # User baru saja klik STOP -> jalankan prediksi otomatis
        if was_playing and not now_playing:
            proc = st.session_state.cam_processor_ref
            if proc is None:
                st.session_state.webcam_result = {"status": "not_detected", "n_frames": 0, "detection_ratio": 0.0, "annotated": None}
            else:
                with proc.lock:
                    raw_seq = np.array(list(proc.feature_buffer), dtype=np.float32)
                    detected_list = list(proc.detected_buffer)
                    last_annotated = proc.last_annotated

                n_frames = len(detected_list)
                detection_ratio = (sum(detected_list) / n_frames) if n_frames else 0.0

                if n_frames == 0 or detection_ratio < MIN_HAND_DETECTION_RATIO:
                    st.session_state.webcam_result = {
                        "status": "not_detected",
                        "n_frames": n_frames,
                        "detection_ratio": detection_ratio,
                        "annotated": last_annotated,
                    }
                else:
                    with st.spinner("Menjalankan prediksi..."):
                        model_input = raw_seq_to_model_input(raw_seq)
                        results, diag = predict_with_ood_check(combined_model, model_input, top_k=5)
                    st.session_state.webcam_result = {
                        "status": "ok" if diag["is_confident"] else "unknown",
                        "results": results,
                        "diag": diag,
                        "n_frames": n_frames,
                        "detection_ratio": detection_ratio,
                        "annotated": last_annotated,
                    }
            st.session_state.cam_processor_ref = None

        st.session_state.cam_was_playing = now_playing

        # ── Tampilkan hasil prediksi terakhir ──────────────────────
        res = st.session_state.webcam_result
        if res is not None:
            status = res["status"]
            extra_caption = (
                f"Rasio deteksi tangan: {res['detection_ratio']*100:.0f}% "
                f"| Frame di buffer: {res['n_frames']}"
            )

            col_res1, col_res2 = st.columns([1, 1])
            with col_res1:
                if res.get("annotated") is not None:
                    st.image(
                        cv2.cvtColor(res["annotated"], cv2.COLOR_BGR2RGB),
                        caption="Frame terakhir + landmark tangan",
                    )

            if status == "not_detected":
                with col_res2:
                    render_badge_card("Tidak Terdeteksi", extra_caption=extra_caption)
                st.warning(
                    "Tangan tidak terdeteksi selama 6 detik rekaman. Klik START lagi dan "
                    "pastikan tangan sepenuhnya terlihat di kamera."
                )
            else:
                diag = res["diag"]
                results = res["results"]
                is_ok = status == "ok"
                badge_label = results[0][0] if is_ok else "Isyarat Tidak Dikenali"

                with col_res2:
                    render_badge_card(badge_label, confidence=diag["confidence"], extra_caption=extra_caption)
                render_diagnostics_caption(diag)

                if not is_ok:
                    st.warning(
                        "Gerakan ini tidak cukup meyakinkan cocok dengan salah satu dari 17 "
                        "kosakata yang dikenali. Berikut kemungkinan terdekat (belum tentu benar):"
                    )

                render_topk_list(results)

                if is_ok:
                    maybe_show_sensitive_note(results[0][0])

        st.caption(
            "Catatan: koneksi webcam menggunakan WebRTC dan server STUN publik. "
            "Pada beberapa jaringan kantor/kampus dengan firewall ketat, koneksi kamera "
            "bisa gagal — gunakan mode Upload Video sebagai alternatif."
        )

# ════════════════════════════════════════════════════════════════
# HALAMAN: TENTANG MODEL
# ════════════════════════════════════════════════════════════════
elif page == "Tentang Model":
    render_topbar("Tentang Model", "Home / Tentang Model")

    section_header("Ringkasan Arsitektur")
    st.markdown(f"""
        <div class="sibi-card">
        <ul>
        <li><b>Arsitektur</b>: {MODEL_INFO['arsitektur']} — 2 attention block,
            2 heads, key_dim=82, dilengkapi Learned Positional Encoding
            (tanpa scaling input √feature_dim — bug pada versi sebelumnya
            yang membuat sinyal urutan waktu nyaris tenggelam, sudah diperbaiki).</li>
        <li><b>Input</b>: sekuens {MODEL_INFO['seq_len']} frame × {MODEL_INFO['feature_dim']}
            fitur (126 koordinat XYZ MediaPipe + 38 fitur geometri tangan).</li>
        <li><b>Loss function</b>: Focal Loss + Label Smoothing (γ=3.0, α=0.25, smoothing=0.1).</li>
        <li><b>Optimizer</b>: Adam dengan Cosine Annealing Warm Restarts (T₀=10).</li>
        <li><b>Dropout</b>: 0.4 (lebih tinggi dari varian tanpa augmentasi, untuk
            mengimbangi kapasitas ekstra dari data yang diperbanyak augmentasi).</li>
        <li><b>Augmentasi data</b>: <i>geometry-consistent augmentation</i> — scaling,
            shift, noise, dan frame-dropout diterapkan pada 126 dim koordinat XYZ
            mentah, lalu ke-38 fitur geometri tangan DIHITUNG ULANG dari koordinat
            yang sudah diaugmentasi (bukan sekadar disalin), supaya fitur geometri
            tetap konsisten secara geometris dengan bentuk tangan hasil augmentasi.</li>
        <li><b>Validasi</b>: 5-fold cross-validation subject-independent,
            3 seed acak per fold (15 run total).</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

    section_header("Hasil Evaluasi (rata-rata 5-fold, 17 kelas)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Akurasi", f"{MODEL_INFO['avg_accuracy']*100:.2f}%")
    col2.metric("F1-Score Macro", f"{MODEL_INFO['avg_f1']*100:.2f}%")
    col3.metric("Std. Akurasi", f"±{MODEL_INFO['std_accuracy']*100:.2f}%")

    section_header("Pipeline Ekstraksi Fitur")
    st.markdown("""
        <div class="sibi-card">
        <ol>
        <li>Ekstraksi landmark 3D tangan (MediaPipe Hands, 21 titik × 2 tangan).</li>
        <li>Pemangkasan frame idle berdasarkan energi gerak.</li>
        <li>Smoothing koordinat Z (moving average) untuk mengurangi noise kedalaman monokuler.</li>
        <li>Resampling temporal ke 45 frame (nearest-neighbor).</li>
        <li>Normalisasi spasial: relatif terhadap pergelangan tangan + skala per tangan.</li>
        <li>Pengayaan 19 fitur geometri per tangan (jarak ujung jari, rasio tekukan jari,
            vektor normal telapak, jarak pinch, keterbukaan tangan).</li>
        </ol>
        </div>
    """, unsafe_allow_html=True)
