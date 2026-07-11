"""
AcousticSpace - Classifier

Production version (per the project brief) fine-tunes a HuggingFace Audio
Spectrogram Transformer (AST) on ASVspoof-style data. That requires network
access to the HuggingFace Hub and the licensed dataset, neither of which is
available in this environment, so this module ships a lightweight CNN that
is architecturally analogous (spectrogram in -> real/fake out) and trains in
seconds on synthetically generated demo data. Swap `AcousticCNN` for an
`ASTForAudioClassification` fine-tune when you have dataset/network access;
the rest of the pipeline (feature extraction, API, frontend) is unaffected.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = os.path.join(os.path.dirname(__file__), "weights", "acoustic_cnn.pt")
FEATURE_DIM = 3 + 3 + 26  # rir(3) + breathing(3) + mfcc mean/std(26)


class AcousticCNN(nn.Module):
    """
    Small 2D-CNN over the mel-spectrogram, fused with the hand-crafted
    RIR/breathing feature vector before the final classification head.
    Mirrors the two-branch idea in the brief: spatial-acoustic features
    (RIR) checked against vocal/spectrogram content.
    """

    def __init__(self, n_mels: int = 80, feature_dim: int = FEATURE_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.spec_fc = nn.Linear(64 * 4 * 4, 64)
        self.feat_fc = nn.Sequential(nn.Linear(feature_dim, 32), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(64 + 32, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 2)
        )

    def forward(self, spec: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        # spec: (B, 1, n_mels, T)  feats: (B, FEATURE_DIM)
        x = self.conv(spec)
        x = x.flatten(1)
        x = F.relu(self.spec_fc(x))
        f = self.feat_fc(feats)
        return self.head(torch.cat([x, f], dim=1))  # logits: [real, fake]


def prep_spectrogram_tensor(mel_db: np.ndarray, target_frames: int = 256) -> torch.Tensor:
    """Pad/crop the mel-spectrogram to a fixed width and normalize."""
    n_mels, t = mel_db.shape
    if t < target_frames:
        mel_db = np.pad(mel_db, ((0, 0), (0, target_frames - t)), mode="constant", constant_values=mel_db.min())
    else:
        mel_db = mel_db[:, :target_frames]
    mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,n_mels,T)


_model_cache = {"model": None}


def load_model() -> AcousticCNN:
    if _model_cache["model"] is not None:
        return _model_cache["model"]

    model = AcousticCNN()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    else:
        # No trained weights yet -- run train_demo_model.py first.
        pass
    model.eval()
    _model_cache["model"] = model
    return model


@torch.no_grad()
def predict(mel_db: np.ndarray, feature_vector: np.ndarray) -> dict:
    model = load_model()
    spec_t = prep_spectrogram_tensor(mel_db)
    feat_t = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)

    logits = model(spec_t, feat_t)
    probs = F.softmax(logits, dim=1).squeeze(0).numpy()

    fake_conf = float(probs[1])
    return {
        "label": "fake" if fake_conf > 0.5 else "real",
        "confidence": round(fake_conf if fake_conf > 0.5 else 1 - fake_conf, 4),
        "fake_probability": round(fake_conf, 4),
        "real_probability": round(float(probs[0]), 4),
    }
