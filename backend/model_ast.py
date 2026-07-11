"""
AcousticSpace - AST Classifier (HuggingFace transformers)

Uses a pretrained Audio Spectrogram Transformer (AST) -- the exact model
family named in the original project brief -- fine-tuned with a 2-class
(real/fake) head. This replaces the earlier CNN stand-in (model.py /
train_demo_model.py, kept in the repo only for reference).

Base checkpoint: MIT/ast-finetuned-audioset-10-10-0.4593
https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593

IMPORTANT -- network requirement:
Loading this model downloads ~340MB of pretrained weights from
huggingface.co the first time it runs. That download only works on a
machine with normal internet access. Run `python train_ast_model.py` once
on your own machine before starting the API server; it caches the
fine-tuned model to weights/ast_finetuned/ so subsequent API starts load
instantly from disk with no network call.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import ASTForAudioClassification, ASTFeatureExtractor

CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
FINE_TUNED_DIR = os.path.join(os.path.dirname(__file__), "weights", "ast_finetuned")
SAMPLE_RATE = 16000
LABEL2ID = {"real": 0, "fake": 1}
ID2LABEL = {0: "real", 1: "fake"}

_cache = {"model": None, "extractor": None}


def is_fine_tuned() -> bool:
    """True only if train_ast_model.py has actually been run and produced
    real fine-tuned weights on disk. If False, load_model() would fall back
    to an untrained classification head whose output is close to random --
    callers should not treat that output as a meaningful signal."""
    return os.path.isdir(FINE_TUNED_DIR)


def load_feature_extractor() -> ASTFeatureExtractor:
    if _cache["extractor"] is None:
        source = FINE_TUNED_DIR if os.path.isdir(FINE_TUNED_DIR) else CHECKPOINT
        _cache["extractor"] = ASTFeatureExtractor.from_pretrained(source)
    return _cache["extractor"]


def load_model() -> ASTForAudioClassification:
    if _cache["model"] is not None:
        return _cache["model"]

    if os.path.isdir(FINE_TUNED_DIR):
        model = ASTForAudioClassification.from_pretrained(FINE_TUNED_DIR)
    else:
        # No fine-tuned checkpoint on disk yet -- fall back to the base
        # pretrained backbone with a freshly initialized 2-class head.
        # Predictions will be near-random until train_ast_model.py is run.
        model = ASTForAudioClassification.from_pretrained(
            CHECKPOINT,
            num_labels=2,
            label2id=LABEL2ID,
            id2label=ID2LABEL,
            ignore_mismatched_sizes=True,
        )
    model.eval()
    _cache["model"] = model
    return model


@torch.no_grad()
def predict(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
    extractor = load_feature_extractor()
    model = load_model()

    inputs = extractor(waveform, sampling_rate=sample_rate, return_tensors="pt")
    logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1).squeeze(0).numpy()

    fake_conf = float(probs[LABEL2ID["fake"]])
    real_conf = float(probs[LABEL2ID["real"]])
    return {
        "label": "fake" if fake_conf > 0.5 else "real",
        "confidence": round(max(fake_conf, real_conf), 4),
        "fake_probability": round(fake_conf, 4),
        "real_probability": round(real_conf, 4),
    }
