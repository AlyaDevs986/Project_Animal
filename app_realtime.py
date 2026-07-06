"""
app_realtime.py
Aplikasi desktop (OpenCV) untuk prediksi realtime dari webcam, video, atau gambar
menggunakan model hasil training_colab.ipynb.

Contoh pakai:
    python app_realtime.py                                   # mode interaktif (ditanya)
    python app_realtime.py --mode webcam
    python app_realtime.py --mode webcam --camera 1
    python app_realtime.py --mode video --source path/to/video.mp4
    python app_realtime.py --mode image --source path/to/image.jpg

Kontrol keyboard (mode webcam/video):
    Q / ESC  -> keluar
    S        -> screenshot & simpan
    P        -> pause / resume
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np

OUTPUT_PATH = "ml_output"
PREDICT_EVERY = 5  # prediksi setiap N frame (biar tidak berat di CPU)


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
def load_model_and_meta():
    meta_path = Path(OUTPUT_PATH) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Tidak menemukan {meta_path}. Pastikan folder '{OUTPUT_PATH}/' "
            "ada di direktori yang sama dengan script ini."
        )

    with open(meta_path, "r") as f:
        meta = json.load(f)

    algo = meta["algorithm"]
    is_dl = meta["is_dl"]

    if is_dl:
        import tensorflow as tf
        model = tf.keras.models.load_model(Path(OUTPUT_PATH) / f"model_{algo}.keras")
        label_encoder = None
    else:
        with open(Path(OUTPUT_PATH) / f"model_{algo}.pkl", "rb") as f:
            model = pickle.load(f)
        with open(Path(OUTPUT_PATH) / "label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)

    return model, meta, label_encoder


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


def predict_frame(frame_bgr, model, meta, label_encoder):
    img_size = tuple(meta["img_size"])
    class_names = meta["class_names"]
    is_dl = meta["is_dl"]

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, img_size)

    if is_dl:
        algo_for_preprocess = meta.get("preprocess") or meta["algorithm"]
        preprocess_input = get_preprocess_fn(algo_for_preprocess)

        x = preprocess_input(resized.astype("float32"))
        x = np.expand_dims(x, axis=0)
        probs = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
    else:
        from skimage.feature import hog
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        feat = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        idx = model.predict([feat])[0]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([feat])[0]
            confidence = float(probs[idx])
        else:
            confidence = None

    label = class_names[idx]
    return label, confidence


def draw_label(frame, label, confidence):
    text = f"{label}" if confidence is None else f"{label} ({confidence*100:.1f}%)"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


# ---------------------------------------------------------------------------
# Mode: gambar tunggal
# ---------------------------------------------------------------------------
def run_image(source, model, meta, label_encoder):
    frame = cv2.imread(source)
    if frame is None:
        print(f"❌ Tidak bisa membaca gambar: {source}")
        return
    label, confidence = predict_frame(frame, model, meta, label_encoder)
    draw_label(frame, label, confidence)
    print(f"🎯 Prediksi: {label}" + (f" ({confidence*100:.1f}%)" if confidence else ""))
    cv2.imshow("Prediksi Gambar - tekan tombol apapun untuk tutup", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Mode: webcam / video (loop realtime)
# ---------------------------------------------------------------------------
def run_stream(cap, model, meta, label_encoder, window_title):
    frame_idx = 0
    paused = False
    label, confidence = "...", None

    print("Kontrol: Q/ESC = keluar | S = screenshot | P = pause/resume")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Stream selesai atau tidak bisa dibaca.")
                break

            if frame_idx % PREDICT_EVERY == 0:
                label, confidence = predict_frame(frame, model, meta, label_encoder)

            frame_idx += 1
            display_frame = frame.copy()
            draw_label(display_frame, label, confidence)
            last_frame = display_frame
        else:
            display_frame = last_frame

        cv2.imshow(window_title, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:  # Q atau ESC
            break
        elif key == ord("s"):
            fname = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(fname, display_frame)
            print(f"📸 Screenshot disimpan: {fname}")
        elif key == ord("p"):
            paused = not paused
            print("⏸️ Paused" if paused else "▶️ Resumed")

    cap.release()
    cv2.destroyAllWindows()


def run_webcam(camera_index, model, meta, label_encoder):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Tidak bisa membuka kamera index {camera_index}. Coba index lain (--camera 1, dst).")
        return
    run_stream(cap, model, meta, label_encoder, "Webcam Realtime - Q untuk keluar")


def run_video(source, model, meta, label_encoder):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Tidak bisa membuka video: {source}")
        return
    run_stream(cap, model, meta, label_encoder, "Video Prediksi - Q untuk keluar")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Realtime image classification (OpenCV)")
    parser.add_argument("--mode", choices=["webcam", "video", "image"], help="Mode operasi")
    parser.add_argument("--source", help="Path video/gambar (untuk mode video/image)")
    parser.add_argument("--camera", type=int, default=0, help="Index kamera (default 0)")
    args = parser.parse_args()

    print("🔄 Memuat model...")
    model, meta, label_encoder = load_model_and_meta()
    print(f"✅ Model '{meta['algorithm']}' dimuat. Kelas: {meta['class_names']}")

    mode = args.mode
    if mode is None:
        print("\n=== MODE INTERAKTIF ===")
        print("1. Webcam")
        print("2. Video")
        print("3. Gambar")
        choice = input("Pilih mode (1-3): ").strip()
        mode = {"1": "webcam", "2": "video", "3": "image"}.get(choice, "webcam")

    if mode == "webcam":
        run_webcam(args.camera, model, meta, label_encoder)
    elif mode == "video":
        source = args.source or input("Path video: ").strip()
        run_video(source, model, meta, label_encoder)
    elif mode == "image":
        source = args.source or input("Path gambar: ").strip()
        run_image(source, model, meta, label_encoder)


if __name__ == "__main__":
    main()
