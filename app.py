"""
Kamus SIBI — Prediksi Isyarat (Model Transformer, Tanpa Augmentasi)
Aplikasi Streamlit untuk mengenali kosakata Bahasa Isyarat Indonesia (SIBI)
dari video unggahan maupun webcam, menggunakan model Transformer 2-Block
yang dilatih pada fitur geometri tangan 164-dimensi (MediaPipe Hands).
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
    WEBCAM_BUFFER_SECONDS, WEBCAM_APPROX_FPS,
)
from utils.preprocessing import (
    preprocess_frames, raw_seq_to_model_input,
    make_live_hands_model, extract_and_annotate,
)
from utils.model_utils import discover_weight_files, load_combined_ensemble_model, predict_single
from utils.ui import inject_global_css, render_topbar, section_header, ACCENT_BLUE

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
        <li>Untuk webcam: klik <b>Mulai Rekam</b>, izinkan akses kamera, lakukan gerakan
            isyarat di depan kamera, lalu klik tombol <b>Prediksi & Hentikan Kamera</b>.</li>
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
            f"⚠️ Tidak ada file bobot (*.weights.h5) di folder `{WEIGHTS_DIR}`. "
            "Letakkan file seperti `Model_2_iter4_seed123.weights.h5` hasil "
            "training Model 2 (Transformer, Tanpa Augmentasi) di folder tersebut."
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
                        results = predict_single(combined_model, model_input, top_k=5)
                        top_label, top_prob = results[0]

                        with col_b:
                            st.markdown(f"""
                                <div class="sibi-card" style="text-align:center;">
                                <p class="sibi-caption">Hasil Prediksi</p>
                                <span class="sibi-result-badge">{top_label}</span>
                                <p style="margin-top:10px;">Keyakinan: <b>{top_prob*100:.1f}%</b></p>
                                <p class="sibi-caption">Rasio deteksi tangan: {info['detection_ratio']*100:.0f}%
                                &nbsp;|&nbsp; Frame asli: {info['n_frame_asli']}</p>
                                </div>
                            """, unsafe_allow_html=True)

                        if info["detection_ratio"] < 0.3:
                            st.warning(
                                "Rasio deteksi tangan rendah — hasil prediksi mungkin kurang "
                                "akurat. Coba video dengan tangan lebih terlihat jelas."
                            )

                        section_header("Top-5 Kemungkinan Kosakata")
                        for label, prob in results:
                            st.write(f"**{label}**")
                            st.progress(min(prob, 1.0), text=f"{prob*100:.1f}%")

                        if top_label in KATA_SENSITIF:
                            st.info(
                                "ℹ️ Kosakata ini termasuk istilah kesehatan reproduksi yang "
                                "digunakan untuk konteks edukasi kesehatan bagi komunitas Tuli."
                            )

    # ── MODE: WEBCAM ─────────────────────────────────────────────
    elif input_mode == "📷 Webcam":
        section_header("Prediksi via Webcam")
        st.markdown(
            '<p class="sibi-caption">Klik <b>Mulai Rekam</b>, izinkan akses kamera, lalu lakukan '
            'gerakan isyarat (kotak hijau & tulisan di video menandakan tangan sudah terdeteksi). '
            'Klik <b>Prediksi & Hentikan Kamera</b> saat selesai — kamera berhenti otomatis dan '
            'hasil langsung muncul.</p>',
            unsafe_allow_html=True,
        )

        buffer_len = WEBCAM_BUFFER_SECONDS * WEBCAM_APPROX_FPS  # fixed, tidak ada opsi

        st.session_state.setdefault("cam_playing", False)
        st.session_state.setdefault("webcam_result", None)

        # ── Video processor: ekstrak landmark & gambar status deteksi ────
        class HandBufferProcessor:
            def __init__(self):
                self.lock = threading.Lock()
                self.feature_buffer = collections.deque(maxlen=buffer_len)
                self.last_annotated = None
                self._hands_model = make_live_hands_model()

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                feat, detected, annotated = extract_and_annotate(
                    img, self._hands_model, draw=True
                )
                with self.lock:
                    self.feature_buffer.append(feat)
                    self.last_annotated = annotated
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        # ctx = webrtc_streamer(
        #     key="sibi-webcam",
        #     mode=WebRtcMode.SENDRECV,
        #     video_processor_factory=HandBufferProcessor,
        #     media_stream_constraints={"video": True, "audio": False},
        #     async_processing=True,
        #     desired_playing_state=st.session_state.cam_playing,
        # )

        ctx = webrtc_streamer(
            key="sibi-webcam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": get_ice_servers()},  # ← TAMBAHKAN BARIS INI
            video_processor_factory=HandBufferProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            desired_playing_state=st.session_state.cam_playing,
        )

        # ── Satu tombol untuk mulai rekam & rekam ulang ──────────────
        if not st.session_state.cam_playing:
            start_label = "🔁 Rekam Ulang" if st.session_state.webcam_result else "▶️ Mulai Rekam"
            if st.button(start_label, type="primary"):
                st.session_state.webcam_result = None
                st.session_state.cam_playing = True
                st.rerun()

        # ── Tombol prediksi, hanya muncul saat kamera aktif ──────────
        if st.session_state.cam_playing:
            predict_clicked = st.button("🔍 Prediksi & Hentikan Kamera", type="primary")

            if predict_clicked:
                if not ctx.video_processor:
                    st.warning("Kamera belum siap. Tunggu sebentar lalu coba lagi.")
                else:
                    with ctx.video_processor.lock:
                        raw_seq = np.array(
                            list(ctx.video_processor.feature_buffer), dtype=np.float32
                        )
                        n_frames = len(ctx.video_processor.feature_buffer)
                        last_annotated = ctx.video_processor.last_annotated

                    if n_frames < 5:
                        st.warning("Buffer belum cukup terisi. Tunggu beberapa detik lalu coba lagi.")
                    else:
                        with st.spinner("Menjalankan prediksi..."):
                            model_input = raw_seq_to_model_input(raw_seq)

                        if model_input is None:
                            st.warning("Belum ada tangan terdeteksi di buffer. Coba lagi.")
                        else:
                            results = predict_single(combined_model, model_input, top_k=5)
                            st.session_state.webcam_result = {
                                "results": results,
                                "n_frames": n_frames,
                                "annotated": last_annotated,
                            }
                            st.session_state.cam_playing = False  # auto-stop kamera
                            st.rerun()

        # ── Tampilkan hasil prediksi terakhir (kamera sudah berhenti) ──────
        if st.session_state.webcam_result is not None:
            res = st.session_state.webcam_result
            top_label, top_prob = res["results"][0]

            col_res1, col_res2 = st.columns([1, 1])
            with col_res1:
                if res["annotated"] is not None:
                    st.image(
                        cv2.cvtColor(res["annotated"], cv2.COLOR_BGR2RGB),
                        caption="Frame terakhir + landmark tangan",
                    )
            with col_res2:
                st.markdown(f"""
                    <div class="sibi-card" style="text-align:center;">
                    <p class="sibi-caption">Hasil Prediksi</p>
                    <span class="sibi-result-badge">{top_label}</span>
                    <p style="margin-top:10px;">Keyakinan: <b>{top_prob*100:.1f}%</b></p>
                    <p class="sibi-caption">Frame di buffer: {res['n_frames']}</p>
                    </div>
                """, unsafe_allow_html=True)

            section_header("Top-5 Kemungkinan Kosakata")
            for label, prob in res["results"]:
                st.write(f"**{label}**")
                st.progress(min(prob, 1.0), text=f"{prob*100:.1f}%")

            if top_label in KATA_SENSITIF:
                st.info(
                    "ℹ️ Kosakata ini termasuk istilah kesehatan reproduksi yang "
                    "digunakan untuk konteks edukasi kesehatan bagi komunitas Tuli."
                )

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
            2 heads, key_dim=82, dilengkapi Learned Positional Encoding.</li>
        <li><b>Input</b>: sekuens {MODEL_INFO['seq_len']} frame × {MODEL_INFO['feature_dim']}
            fitur (126 koordinat XYZ MediaPipe + 38 fitur geometri tangan).</li>
        <li><b>Loss function</b>: Focal Loss + Label Smoothing (γ=3.0, α=0.25, smoothing=0.1).</li>
        <li><b>Optimizer</b>: Adam dengan Cosine Annealing Warm Restarts (T₀=10).</li>
        <li><b>Augmentasi data</b>: tidak digunakan — dinonaktifkan karena secara
            empiris menurunkan performa varian Transformer.</li>
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
