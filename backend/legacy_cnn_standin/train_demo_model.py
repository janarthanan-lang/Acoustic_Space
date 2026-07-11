"""
AcousticSpace - Demo model training

Generates synthetic "real" (reverberant, natural breath gaps) and "fake"
(dry / mismatched reverb, uniform or missing breath gaps) audio samples,
then trains AcousticCNN on them. This exists purely so the pipeline runs
end-to-end without needing network access to ASVspoof / HuggingFace.

Swap this for real fine-tuning on ASVspoof + a HuggingFace AST checkpoint
once you have network/dataset access -- see model.py docstring.

Run:  python train_demo_model.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from audio_processing import (
    extract_mel_spectrogram, estimate_rir_envelope,
    extract_breathing_pattern, SAMPLE_RATE,
)
from model import AcousticCNN, prep_spectrogram_tensor, MODEL_PATH, FEATURE_DIM

RNG = np.random.default_rng(42)


def _synthesize_voice_like(duration=3.0, sr=SAMPLE_RATE, reverb_amount=0.0,
                            breath_regular=False, n_phrases=4):
    """Build a toy speech-like signal: formant-ish tones with amplitude
    envelopes, separated by breath gaps, optionally convolved with a
    synthetic room impulse response."""
    total_samples = int(duration * sr)
    y = np.zeros(total_samples)
    phrase_len = total_samples // n_phrases

    pos = 0
    for p in range(n_phrases):
        seg_len = int(phrase_len * RNG.uniform(0.55, 0.8))
        t = np.arange(seg_len) / sr
        f0 = RNG.uniform(90, 220)
        sig = (
            0.6 * np.sin(2 * np.pi * f0 * t)
            + 0.25 * np.sin(2 * np.pi * f0 * 2.2 * t)
            + 0.1 * RNG.standard_normal(seg_len)
        )
        env = np.hanning(seg_len)
        sig = sig * env
        end = min(pos + seg_len, total_samples)
        y[pos:end] += sig[: end - pos]

        gap = phrase_len - seg_len
        if breath_regular:
            gap = int(phrase_len * 0.2)  # unnaturally uniform gap
        else:
            gap = int(gap * RNG.uniform(0.6, 1.4))
        pos = min(pos + seg_len + gap, total_samples)

    if reverb_amount > 0:
        rt = RNG.uniform(0.15, 0.6)
        ir_len = int(rt * sr)
        decay = np.exp(-np.linspace(0, 6, ir_len))
        ir = decay * RNG.standard_normal(ir_len)
        ir /= np.abs(ir).max() + 1e-6
        y = np.convolve(y, reverb_amount * ir, mode="full")[:total_samples]

    y = y / (np.abs(y).max() + 1e-6) * 0.8
    return y.astype(np.float32)


class DemoDataset(Dataset):
    def __init__(self, n_samples=240):
        self.samples = []
        for i in range(n_samples):
            is_fake = i % 2 == 0
            if is_fake:
                # dry synthesis, no/uniform breathing -> "fake"
                y = _synthesize_voice_like(
                    reverb_amount=RNG.choice([0.0, 0.05]), breath_regular=True
                )
            else:
                # natural reverb + irregular breathing -> "real"
                y = _synthesize_voice_like(
                    reverb_amount=RNG.uniform(0.3, 0.7), breath_regular=False
                )
            self.samples.append((y, int(is_fake)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        y, label = self.samples[idx]
        mel = extract_mel_spectrogram(y)
        rir = estimate_rir_envelope(y)
        breath = extract_breathing_pattern(y)
        import librosa
        mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=13)
        mfcc_stats = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
        feats = np.concatenate([list(rir.values()), list(breath.values()), mfcc_stats]).astype(np.float32)

        spec_t = prep_spectrogram_tensor(mel).squeeze(0)  # (1, n_mels, T)
        return spec_t, torch.tensor(feats, dtype=torch.float32), label


def train(epochs=8, batch_size=16, lr=1e-3):
    print("Generating synthetic demo dataset...")
    dataset = DemoDataset(n_samples=240)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AcousticCNN(feature_dim=FEATURE_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for spec, feats, labels in loader:
            opt.zero_grad()
            logits = model(spec, feats)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
        print(f"Epoch {epoch+1}/{epochs}  loss={total_loss/total:.4f}  acc={correct/total:.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved demo model to {MODEL_PATH}")


if __name__ == "__main__":
    train()
