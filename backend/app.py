import os
import io
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 320
TYPE_NAMES = ["real", "face_swap", "face_reenactment", "full_ai_generated"]

# ---------------------------------------------------------------------------
# IMAGE MODEL — unchanged from your original file
# ---------------------------------------------------------------------------
class DeepfakeNet(smp.Unet):
    def __init__(self, n_types=4, **kwargs):
        super().__init__(**kwargs)
        enc_channels = self.encoder.out_channels[-1]
        self.type_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(enc_channels, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, n_types)
        )

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        mask_logits = self.segmentation_head(decoder_output)
        cls_logit = self.classification_head(features[-1])
        type_logits = self.type_head(features[-1])
        return mask_logits, cls_logit, type_logits


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "best_model.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading image model...")
model = DeepfakeNet(
    n_types=4,
    encoder_name="timm-efficientnet-b4",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    aux_params=dict(pooling="avg", dropout=0.3, classes=1),
).to(device)

state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()
print("Image model loaded.")

val_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2(),
])

app = Flask(__name__)
CORS(app)


@torch.no_grad()
def run_inference(pil_image: Image.Image, threshold=0.5):
    img = np.array(pil_image.convert("RGB"))
    inp = val_tf(image=img)["image"].unsqueeze(0).to(device)

    mask_logits, cls_logit, type_logits = model(inp)
    fake_prob = torch.sigmoid(cls_logit).item()
    type_probs = torch.softmax(type_logits, dim=1).squeeze().cpu().numpy()
    pred_type = TYPE_NAMES[int(type_probs.argmax())]

    if fake_prob > threshold:
        result = "AI-Generated" if pred_type == "full_ai_generated" else "Deepfake"
        confidence = fake_prob
    else:
        result = "Real"
        confidence = 1 - fake_prob

    return result, confidence, pred_type


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        image = Image.open(io.BytesIO(file.read()))
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    try:
        result, confidence, fake_type = run_inference(image)
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Could not process image"}), 400

    return jsonify({
        "result": result,
        "confidence": confidence,
        "fake_type": fake_type
    })


# ============================================================
# VIDEO MODEL — new, fully separate from the image model above
# ============================================================
import cv2
import tempfile
import librosa
from video_model import FusionModel

VIDEO_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "video_model.pt")
VIDEO_FRAMES_PER_CLIP = 20
VIDEO_LABEL_MAP = {0: "Real", 1: "Deepfake"}

print("Loading video model...")
video_model = FusionModel().to(device)
video_state_dict = torch.load(VIDEO_MODEL_PATH, map_location=device, weights_only=True)
video_model.load_state_dict(video_state_dict)
video_model.eval()
print("Video model loaded.")


def extract_faces_for_video_model(video_path, n_frames=VIDEO_FRAMES_PER_CLIP, size=224):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    idxs = np.linspace(0, total - 1, min(n_frames, total)).astype(int)
    faces = []
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in idxs:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (size, size))
            faces.append(rgb)
    cap.release()
    if len(faces) == 0:
        return None
    while len(faces) < n_frames:
        faces.append(faces[-1])
    faces = np.stack(faces[:n_frames]).astype(np.float32) / 255.0
    return faces


def extract_audio_for_video_model(video_path, sr=16000, duration=4.0):
    target_len = int(sr * duration)
    try:
        wav, _ = librosa.load(video_path, sr=sr, mono=True, duration=duration)
        if wav is None or len(wav) == 0:
            raise ValueError("empty audio")
    except Exception:
        wav = np.zeros(target_len, dtype=np.float32)
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    else:
        wav = wav[:target_len]
    return wav.astype(np.float32)


@torch.no_grad()
def run_video_inference(video_path):
    faces = extract_faces_for_video_model(video_path)
    if faces is None:
        raise ValueError("No frames could be read from video")
    wav = extract_audio_for_video_model(video_path)

    frames_t = torch.from_numpy(faces).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    wav_t = torch.from_numpy(wav).unsqueeze(0).to(device)

    f_logits, _, _ = video_model(frames_t, wav_t)
    probs = torch.softmax(f_logits, dim=1).cpu().numpy()[0]
    pred_idx = int(probs.argmax())
    result = VIDEO_LABEL_MAP[pred_idx]
    confidence = float(probs[pred_idx])
    return result, confidence


@app.route("/predict_video", methods=["POST"])
def predict_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        result, confidence = run_video_inference(tmp_path)

        return jsonify({
            "result": result,
            "confidence": confidence
        })

    except Exception as e:
        print("Video prediction error:", e)
        return jsonify({"error": "Could not process video"}), 400
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)