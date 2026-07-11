"""
AcousticSpace - API Gateway (FastAPI)
Serves the ML model for low-latency inference. Accepts an audio upload,
runs the acoustic pipeline, and returns spectrogram + RIR/breathing
features + real/fake classification for the React dashboard.

Verdict logic (important): the fake-probability is a HYBRID of two signals:
  1. `heuristic_fake_score` -- deterministic, physics-based, computed purely
     from the RIR/breathing features. Always available, never depends on
     training data quality.
  2. The fine-tuned AST model's prediction -- only used if
     `model_ast.is_fine_tuned()` is True. If you haven't run
     train_ast_model.py yet, the AST model is skipped entirely rather than
     trusting its untrained (near-random) head.
This two-signal design is deliberate: a small AST fine-tuned on synthetic
demo data won't generalize perfectly to real-world speech, so letting it
override real physical evidence (a genuine room echo, natural breathing)
was producing false "fake" verdicts on real recordings. Blending keeps the
system honest at every stage of training maturity.
"""

import os
import shutil
import tempfile
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from audio_processing import analyze_audio, load_audio, heuristic_fake_score
import model_ast

app = FastAPI(title="AcousticSpace API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# .webm/.ogg included because browser MediaRecorder (live mic capture)
# outputs webm/opus-encoded audio, not wav.
ALLOWED_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
MAX_FILE_MB = 25

# Ensemble weights when the AST model IS fine-tuned. Heuristic is weighted
# higher because it's deterministic physics, not a model trained on a few
# hundred synthetic clips.
HEURISTIC_WEIGHT = 0.55
MODEL_WEIGHT = 0.45

# in-memory history for the "analysis history" feature (swap for a DB in prod)
_history: list[dict] = []


class AnalysisResponse(BaseModel):
    id: str
    filename: str
    duration_sec: float
    label: str
    confidence: float
    fake_probability: float
    real_probability: float
    rir_features: dict
    breathing_features: dict
    mel_spectrogram: list
    verdict_reason: str
    model_status: str  # "fine-tuned" | "heuristic_only"


def _build_verdict_reason(rir: dict, breath: dict, label: str, model_status: str) -> str:
    if label == "fake":
        reasons = []
        if rir["rt60_estimate_sec"] < 0.06:
            reasons.append("acoustic reflections are almost absent (unnaturally dry room signature)")
        if breath["breath_gap_count"] >= 2 and breath["breath_gap_regularity"] > 0.88:
            reasons.append("breathing gaps are suspiciously uniform for natural speech")
        if not reasons:
            reasons.append("combined acoustic/breathing evidence leans synthetic")
        suffix = "" if model_status == "fine-tuned" else " (heuristic-only mode -- run train_ast_model.py to add the transformer model)"
        return "Flagged as synthetic: " + "; ".join(reasons) + "." + suffix
    suffix = "" if model_status == "fine-tuned" else " (heuristic-only mode -- run train_ast_model.py to add the transformer model)"
    return "Acoustic reflections and breathing cadence are consistent with a genuine single-environment recording." + suffix


@app.get("/health")
def health():
    return {"status": "ok", "ast_fine_tuned": model_ast.is_fine_tuned()}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXT)}")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    if os.path.getsize(tmp_path) > MAX_FILE_MB * 1024 * 1024:
        os.remove(tmp_path)
        raise HTTPException(400, f"File exceeds {MAX_FILE_MB}MB limit")

    try:
        analysis = analyze_audio(tmp_path)
        rir = analysis["rir_features"]
        breath = analysis["breathing_features"]

        h_score = heuristic_fake_score(rir, breath)  # always computed

        if model_ast.is_fine_tuned():
            y = load_audio(tmp_path)
            ast_result = model_ast.predict(waveform=y)
            fake_prob = HEURISTIC_WEIGHT * h_score + MODEL_WEIGHT * ast_result["fake_probability"]
            model_status = "fine-tuned"
        else:
            fake_prob = h_score
            model_status = "heuristic_only"
    finally:
        os.remove(tmp_path)

    fake_prob = max(0.03, min(0.97, fake_prob))
    real_prob = 1 - fake_prob
    label = "fake" if fake_prob > 0.5 else "real"
    confidence = round(max(fake_prob, real_prob), 4)

    record = {
        "id": str(uuid.uuid4())[:8],
        "filename": file.filename,
        "duration_sec": analysis["duration_sec"],
        "label": label,
        "confidence": confidence,
        "fake_probability": round(fake_prob, 4),
        "real_probability": round(real_prob, 4),
        "rir_features": rir,
        "breathing_features": breath,
        "mel_spectrogram": analysis["mel_spectrogram"],
        "verdict_reason": _build_verdict_reason(rir, breath, label, model_status),
        "model_status": model_status,
    }
    _history.append({k: v for k, v in record.items() if k != "mel_spectrogram"})
    return record


@app.get("/history")
def get_history():
    return list(reversed(_history[-50:]))
