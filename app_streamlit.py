"""
app_streamlit.py
Web UI (Streamlit) untuk klasifikasi gambar/video/webcam
menggunakan model yang sudah dilatih dari training_colab.ipynb.

Jalankan dengan:
    streamlit run app_streamlit.py
"""

import os
import json
import pickle
from pathlib import Path

import numpy as np
import cv2
import streamlit as st
from PIL import Image

OUTPUT_PATH = "ml_output"  # folder hasil training (harus ada di direktori yang sama)

# Warna khas per kelas — dipakai di kartu hasil & bar probabilitas.
# Kalau nama kelas kamu beda, tambahkan saja di sini; kelas yang tidak
# terdaftar otomatis dapat warna dari FALLBACK_PALETTE secara berurutan.
CLASS_COLORS = {
    "dogs": "#C97B3D",
    "dog": "#C97B3D",
    "cats": "#6C63A6",
    "cat": "#6C63A6",
    "panda": "#3A473A",
}
FALLBACK_PALETTE = ["#3F6444", "#B0562B", "#4C6E8A", "#8A4C6E", "#6E8A4C"]


def color_for_class(name, index):
    return CLASS_COLORS.get(name.lower(), FALLBACK_PALETTE[index % len(FALLBACK_PALETTE)])


# ---------------------------------------------------------------------------
# Custom styling — mengganti tampilan default Streamlit yang polos
# ---------------------------------------------------------------------------
def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700;900&family=Work+Sans:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Work Sans', sans-serif;
        }

        .stApp {
            background: linear-gradient(180deg, #EFF3EA 0%, #E7EDE1 100%);
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* ---------- Hero banner ---------- */
        .hero-banner {
            background: #22301F;
            border-radius: 20px;
            padding: 2.4rem 2.2rem;
            margin-bottom: 1.6rem;
            color: #F5F1E6;
            position: relative;
            overflow: hidden;
        }
        .hero-banner::after {
            content: "🐾";
            position: absolute;
            right: -10px;
            bottom: -30px;
            font-size: 9rem;
            opacity: 0.08;
        }
        .hero-eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.75rem;
            font-weight: 600;
            color: #B7C4AC;
            margin-bottom: 0.5rem;
        }
        .hero-banner h1 {
            font-family: 'Fraunces', serif;
            font-weight: 900;
            font-size: 2.3rem;
            margin: 0 0 0.5rem 0;
            letter-spacing: -0.02em;
            line-height: 1.15;
        }
        .hero-banner p {
            font-size: 0.98rem;
            opacity: 0.85;
            margin: 0;
            max-width: 520px;
        }
        .hero-badges {
            display: flex;
            gap: 0.6rem;
            margin-top: 1.3rem;
            flex-wrap: wrap;
        }
        .hero-badge {
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.25);
            padding: 0.32rem 0.95rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 500;
        }

        /* ---------- Tabs ---------- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            background: #FFFFFF;
            border-radius: 12px 12px 0 0;
            padding: 0.65rem 1.3rem;
            font-weight: 600;
            color: #4B5A45;
            border: 1px solid #DDE3D4;
            border-bottom: none;
        }
        .stTabs [aria-selected="true"] {
            background: #22301F !important;
            color: #F5F1E6 !important;
        }

        /* ---------- Result card ---------- */
        .result-card {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 1.5rem 1.7rem;
            border: 1px solid #E1E7D9;
            box-shadow: 0 2px 12px rgba(34,48,31,0.07);
            border-left: 6px solid var(--accent, #3F6444);
        }
        .result-kicker {
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-size: 0.72rem;
            font-weight: 600;
            color: #8A9483;
            margin-bottom: 0.3rem;
        }
        .result-label {
            font-family: 'Fraunces', serif;
            font-weight: 700;
            font-size: 1.7rem;
            margin: 0 0 1rem 0;
            text-transform: capitalize;
            color: #22301F;
        }
        .result-label span {
            font-family: 'Work Sans', sans-serif;
            font-weight: 600;
            font-size: 1rem;
            color: #6B7566;
            margin-left: 0.5rem;
        }
        .prob-row {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            margin-bottom: 0.6rem;
        }
        .prob-name {
            width: 78px;
            flex-shrink: 0;
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: capitalize;
            color: #3A473A;
        }
        .prob-track {
            flex: 1;
            background: #EEF1E8;
            border-radius: 999px;
            height: 13px;
            overflow: hidden;
        }
        .prob-fill {
            height: 100%;
            border-radius: 999px;
            transition: width 0.4s ease;
        }
        .prob-pct {
            width: 48px;
            flex-shrink: 0;
            text-align: right;
            font-size: 0.8rem;
            font-weight: 600;
            color: #6B7566;
        }

        /* ---------- File uploader ---------- */
        [data-testid="stFileUploaderDropzone"] {
            background: #FFFFFF !important;
            border: 2px dashed #C7D0BB !important;
            border-radius: 14px !important;
        }

        /* ---------- Misc ---------- */
        .section-caption {
            color: #5C6A56;
            font-size: 0.92rem;
            margin-bottom: 0.8rem;
        }
        hr {
            border-color: #DDE3D4 !important;
        }

        /* ---------- Floating background decorations ---------- */
        .floating-bg {
            position: fixed;
            inset: 0;
            overflow: hidden;
            z-index: 0;
            pointer-events: none;
        }
        .floater {
            position: absolute;
            bottom: -10vh;
            font-size: 2.1rem;
            opacity: 0;
            animation-name: floatUp;
            animation-timing-function: ease-in;
            animation-iteration-count: infinite;
            filter: drop-shadow(0 2px 4px rgba(34,48,31,0.08));
        }
        @keyframes floatUp {
            0%   { transform: translateY(0) rotate(-8deg) scale(0.9); opacity: 0; }
            10%  { opacity: 0.28; }
            50%  { transform: translateY(-55vh) rotate(6deg) scale(1.05); }
            90%  { opacity: 0.22; }
            100% { transform: translateY(-115vh) rotate(-6deg) scale(0.95); opacity: 0; }
        }
        @media (prefers-reduced-motion: reduce) {
            .floater { animation: none; opacity: 0.15; }
        }

        /* Make real content sit above the floating layer */
        .block-container {
            position: relative;
            z-index: 1;
        }

        /* ---------- Entrance animations ---------- */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(14px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .hero-banner {
            animation: fadeInUp 0.6s ease both;
        }
        .result-card {
            animation: fadeInUp 0.45s ease both;
        }

        /* Gentle hover lift on uploaded image previews */
        [data-testid="stImage"] img {
            border-radius: 14px;
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }
        [data-testid="stImage"] img:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(34,48,31,0.15);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_floating_background():
    """Dekorasi latar belakang: jejak kaki & ikon hewan yang melayang pelan.
    Murni CSS animation, tidak mengganggu performa atau interaksi (pointer-events: none).
    """
    floaters = [
        ("🐾", "4%", 22, 0),
        ("🐶", "16%", 26, 3),
        ("🐾", "28%", 19, 7),
        ("🐼", "42%", 30, 1),
        ("🐾", "55%", 23, 9),
        ("🐱", "68%", 27, 4),
        ("🐾", "80%", 21, 11),
        ("🐾", "91%", 25, 6),
    ]
    items = "".join(
        f'<span class="floater" style="left:{left}; animation-duration:{dur}s; animation-delay:{delay}s;">{emoji}</span>'
        for emoji, left, dur, delay in floaters
    )
    st.markdown(f'<div class="floating-bg">{items}</div>', unsafe_allow_html=True)


def render_hero(meta):
    classes_str = " &nbsp;·&nbsp; ".join(c.capitalize() for c in meta["class_names"])
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-eyebrow">Model Klasifikasi Gambar</div>
            <h1>🐾 Animal Classifier</h1>
            <p>Kenali jenis hewan dari foto, video, atau webcam secara otomatis
            menggunakan model {meta['algorithm']}.</p>
            <div class="hero-badges">
                <div class="hero-badge">🏷️ {classes_str}</div>
                <div class="hero-badge">🎯 Akurasi test {meta['accuracy']*100:.1f}%</div>
                <div class="hero-badge">⚙️ {meta['algorithm']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(label, confidence, prob_dict, class_names):
    accent = color_for_class(label, class_names.index(label) if label in class_names else 0)
    conf_text = f"<span>({confidence*100:.1f}% yakin)</span>" if confidence is not None else ""

    rows_html = ""
    if prob_dict:
        for i, cls in enumerate(class_names):
            p = prob_dict.get(cls, 0.0)
            c = color_for_class(cls, i)
            rows_html += f"""
            <div class="prob-row">
                <div class="prob-name">{cls}</div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{p*100:.1f}%; background:{c};"></div>
                </div>
                <div class="prob-pct">{p*100:.1f}%</div>
            </div>
            """

    st.markdown(
        f"""
        <div class="result-card" style="--accent:{accent}">
            <div class="result-kicker">Hasil Prediksi</div>
            <div class="result-label">{label} {conf_text}</div>
            {rows_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Load model & metadata (sekali saja, di-cache biar tidak reload terus)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_and_meta():
    meta_path = Path(OUTPUT_PATH) / "metadata.json"
    if not meta_path.exists():
        return None, None, None

    with open(meta_path, "r") as f:
        meta = json.load(f)

    algo = meta["algorithm"]
    is_dl = meta["is_dl"]

    if is_dl:
        import tensorflow as tf
        model_path = Path(OUTPUT_PATH) / f"model_{algo}.keras"
        model = tf.keras.models.load_model(model_path)
        extra = None
    else:
        with open(Path(OUTPUT_PATH) / f"model_{algo}.pkl", "rb") as f:
            model = pickle.load(f)
        with open(Path(OUTPUT_PATH) / "label_encoder.pkl", "rb") as f:
            extra = pickle.load(f)  # label encoder untuk classical ML

    return model, meta, extra


def get_preprocess_fn(algo_name):
    """Ambil fungsi preprocess_input yang benar sesuai arsitektur pretrained.
    PENTING: harus sama persis dengan yang dipakai saat training (Cell 6 & 10
    di training_colab.ipynb) — bukan asal dibagi 255.
    """
    if algo_name == "MobileNetV2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    elif algo_name == "ResNet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
    elif algo_name == "EfficientNetB0":
        from tensorflow.keras.applications.efficientnet import preprocess_input
    else:
        raise ValueError(f"Algoritma DL tidak dikenal: {algo_name}")
    return preprocess_input


def predict_image(img_rgb, model, meta, label_encoder):
    """img_rgb: numpy array RGB (H, W, 3) -> return (label, confidence, dict probs)"""
    img_size = tuple(meta["img_size"])
    class_names = meta["class_names"]
    is_dl = meta["is_dl"]

    resized = cv2.resize(img_rgb, img_size)

    if is_dl:
        algo_for_preprocess = meta.get("preprocess") or meta["algorithm"]
        preprocess_input = get_preprocess_fn(algo_for_preprocess)

        x = preprocess_input(resized.astype("float32"))
        x = np.expand_dims(x, axis=0)
        probs = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = class_names[idx]
        confidence = float(probs[idx])
        prob_dict = {c: float(p) for c, p in zip(class_names, probs)}
    else:
        from skimage.feature import hog
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        feat = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        idx = model.predict([feat])[0]
        label = class_names[idx]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([feat])[0]
            confidence = float(probs[idx])
            prob_dict = {c: float(p) for c, p in zip(class_names, probs)}
        else:
            confidence = None
            prob_dict = {}

    return label, confidence, prob_dict


# ---------------------------------------------------------------------------
# UI utama
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Animal Classifier", page_icon="🐾", layout="centered")
inject_custom_css()
render_floating_background()

model, meta, label_encoder = load_model_and_meta()

if model is None:
    st.error(
        f"Model belum ditemukan. Pastikan folder `{OUTPUT_PATH}/` ada di direktori "
        "yang sama dengan file ini, hasil dari training_colab.ipynb."
    )
    st.stop()

render_hero(meta)

tab_image, tab_video, tab_webcam = st.tabs(["📷 Upload Gambar", "🎞️ Upload Video", "🔴 Webcam Realtime"])

# --- TAB 1: Upload gambar (bisa banyak sekaligus) ---
with tab_image:
    st.markdown('<p class="section-caption">Upload satu atau beberapa foto untuk diprediksi.</p>', unsafe_allow_html=True)
    files = st.file_uploader(
        "Pilih gambar", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if files:
        for file in files:
            img = Image.open(file).convert("RGB")
            img_arr = np.array(img)
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(img, caption=file.name, use_container_width=True)
            with col2:
                label, confidence, prob_dict = predict_image(img_arr, model, meta, label_encoder)
                render_result_card(label, confidence, prob_dict, meta["class_names"])
            st.divider()

# --- TAB 2: Upload video, prediksi per beberapa frame ---
with tab_video:
    st.markdown('<p class="section-caption">Upload video, akan diprediksi setiap N frame agar tidak terlalu berat.</p>', unsafe_allow_html=True)
    every_n = st.slider("Prediksi setiap berapa frame?", 1, 30, 10)
    video_file = st.file_uploader("Pilih video", type=["mp4", "avi", "mov", "mkv"], label_visibility="collapsed")

    if video_file is not None:
        tmp_path = f"_tmp_{video_file.name}"
        with open(tmp_path, "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture(tmp_path)
        frame_idx = 0
        frame_placeholder = st.empty()
        result_placeholder = st.empty()
        progress = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % every_n == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                label, confidence, prob_dict = predict_image(rgb, model, meta, label_encoder)
                frame_placeholder.image(rgb, caption=f"Frame {frame_idx}", use_container_width=True)
                with result_placeholder.container():
                    render_result_card(label, confidence, prob_dict, meta["class_names"])
            frame_idx += 1
            progress.progress(min(frame_idx / total_frames, 1.0))

        cap.release()
        os.remove(tmp_path)
        st.success("✅ Selesai memproses video.")

# --- TAB 3: Webcam realtime (streamlit-webrtc, fallback ke snapshot) ---
with tab_webcam:
    st.markdown('<p class="section-caption">Prediksi langsung dari webcam.</p>', unsafe_allow_html=True)
    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
        import av

        class Predictor(VideoProcessorBase):
            def __init__(self):
                self.label = ""
                self.confidence = 0.0

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                label, confidence, _ = predict_image(rgb, model, meta, label_encoder)
                self.label, self.confidence = label, confidence or 0.0

                text = f"{label} ({self.confidence*100:.1f}%)"
                cv2.rectangle(img, (0, 0), (img.shape[1], 40), (34, 48, 31), -1)
                cv2.putText(img, text, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 241, 230), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(key="realtime", video_processor_factory=Predictor)
        st.caption("Jika kamera tidak muncul, izinkan akses kamera di browser.")

    except ImportError:
        st.warning(
            "`streamlit-webrtc` belum terinstall. Jalankan `pip install streamlit-webrtc av` "
            "untuk mode webcam realtime (kamera nyala terus, tanpa perlu ambil foto manual). "
            "Sementara ini pakai mode snapshot di bawah."
        )
        snapshot = st.camera_input("Ambil foto dari webcam")
        if snapshot is not None:
            img = Image.open(snapshot).convert("RGB")
            img_arr = np.array(img)
            label, confidence, prob_dict = predict_image(img_arr, model, meta, label_encoder)
            render_result_card(label, confidence, prob_dict, meta["class_names"])