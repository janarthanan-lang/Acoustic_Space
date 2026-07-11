"""
AcousticSpace - Audio Processing Pipeline
Extracts low-level acoustic features, isolating Room Impulse Response (RIR)
and environmental reverb characteristics from raw audio, plus breathing /
silence-gap cadence used to cross-check against spoken syllable timing.

Core idea: a genuine recording carries a *consistent* acoustic fingerprint of
the room it was captured in (reverberation time, spectral decay envelope,
noise floor). Generative TTS/voice-clone audio is usually rendered "dry" or
pasted onto unrelated background noise, so the fingerprint is inconsistent
across the clip or mismatched with the claimed environment.
"""

import numpy as np
import librosa
import librosa.display


SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def extract_mel_spectrogram(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Log-mel spectrogram - the base representation fed to the transformer."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    return librosa.power_to_db(mel, ref=np.max)


def estimate_rir_envelope(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """
    Approximate Room Impulse Response characteristics without a reference
    impulse, using the decay envelope of onset-triggered energy (a
    blind-RIR proxy widely used when a clean reference signal isn't
    available). Returns reverberation-time estimate (RT60-like) and a
    spectral decay slope, which together behave as a lightweight
    fingerprint of the recording environment.
    """
    stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    energy = np.sum(stft ** 2, axis=0)
    energy_db = 10 * np.log10(energy + 1e-10)

    # Find decay segments: energy peak -> where it drops 20dB (proxy T20 -> RT60)
    decay_times = []
    i = 0
    frame_time = HOP_LENGTH / sr
    while i < len(energy_db) - 1:
        if energy_db[i] > np.percentile(energy_db, 75):
            peak = energy_db[i]
            j = i
            while j < len(energy_db) - 1 and energy_db[j] > peak - 20:
                j += 1
            decay_times.append((j - i) * frame_time)
            i = j
        i += 1

    rt60_estimate = float(np.median(decay_times)) * 3 if decay_times else 0.0
    # Real rooms rarely exceed ~3s RT60; silence-heavy clips can otherwise
    # blow this estimate up to nonsensical values (seen: 170s on a clip with
    # long pauses). Clamp to a physically plausible range.
    rt60_estimate = float(np.clip(rt60_estimate, 0.0, 3.0))

    # Spectral decay slope: how fast high frequencies die relative to low
    # frequencies after onsets -- rooms absorb highs faster than lows.
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    band_low = stft[freqs < 1000].mean(axis=0)
    band_high = stft[freqs > 4000].mean(axis=0)
    ratio = np.log((band_high + 1e-6) / (band_low + 1e-6))
    spectral_decay_slope = float(np.std(ratio))

    return {
        "rt60_estimate_sec": round(rt60_estimate, 4),
        "spectral_decay_slope": round(spectral_decay_slope, 4),
        "noise_floor_db": round(float(np.percentile(energy_db, 5)), 2),
    }


def extract_breathing_pattern(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """
    Detects low-energy, low-frequency gaps between voiced segments that
    correspond to breathing. Real speech has irregular but physiologically
    plausible breath gaps aligned with phrase boundaries; many TTS/clone
    pipelines omit breathing entirely or insert unnaturally uniform gaps.
    """
    intervals = librosa.effects.split(y, top_db=30)
    if len(intervals) < 2:
        return {"breath_gap_count": 0, "breath_gap_regularity": 0.0, "mean_gap_sec": 0.0}

    gaps = []
    for k in range(len(intervals) - 1):
        gap_samples = intervals[k + 1][0] - intervals[k][1]
        gap_sec = gap_samples / sr
        if 0.05 < gap_sec < 1.2:  # plausible breath-gap range
            gaps.append(gap_sec)

    if not gaps:
        return {"breath_gap_count": 0, "breath_gap_regularity": 0.0, "mean_gap_sec": 0.0}

    gaps = np.array(gaps)
    # Regularity close to 1.0 = suspiciously uniform gaps (machine-like)
    regularity = 1.0 - min(float(np.std(gaps) / (np.mean(gaps) + 1e-6)), 1.0)

    return {
        "breath_gap_count": int(len(gaps)),
        "breath_gap_regularity": round(regularity, 4),
        "mean_gap_sec": round(float(np.mean(gaps)), 4),
    }


def extract_feature_vector(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Flat numeric feature vector combining RIR + breathing features, used by
    the lightweight classifier. The transformer branch (model.py) instead
    consumes the full mel-spectrogram.
    """
    rir = estimate_rir_envelope(y, sr)
    breath = extract_breathing_pattern(y, sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_stats = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])

    vec = np.concatenate([
        list(rir.values()),
        list(breath.values()),
        mfcc_stats,
    ]).astype(np.float32)
    return vec


def heuristic_fake_score(rir: dict, breath: dict) -> float:
    """
    Deterministic, physics-based fake-probability estimate derived purely
    from the RIR + breathing features above -- no ML, no training data
    needed. This is the trustworthy fallback (and always a contributing
    signal) because it doesn't depend on how well a small fine-tuned model
    generalizes to real-world audio.

    Thresholds are intentionally conservative: a clip needs a clear
    physical anomaly (near-zero reverb, near-metronomic breathing) to be
    pushed toward "fake" -- ambiguous evidence stays close to "real".
    """
    score = 0.25  # baseline: mildly presume real when evidence is ambiguous

    rt60 = rir["rt60_estimate_sec"]
    if rt60 < 0.06:
        score += 0.30  # near-zero reverb: suspiciously dry / studio-clean synthesis

    gap_count = breath["breath_gap_count"]
    regularity = breath["breath_gap_regularity"]
    if gap_count >= 2 and regularity > 0.88:
        score += 0.35  # near-metronomic breathing -- classic TTS tell
    elif gap_count == 0:
        score += 0.05  # no detectable breathing at all -- weak signal only

    if rir["noise_floor_db"] < -95:
        score += 0.05  # digitally near-silent floor, mildly synthetic-leaning

    return float(np.clip(score, 0.03, 0.97))


def analyze_audio(path: str) -> dict:
    """Full pipeline entry point used by the API layer."""
    y = load_audio(path)
    duration = len(y) / SAMPLE_RATE

    mel = extract_mel_spectrogram(y)
    rir = estimate_rir_envelope(y)
    breathing = extract_breathing_pattern(y)
    feature_vector = extract_feature_vector(y)

    return {
        "duration_sec": round(duration, 3),
        "mel_spectrogram_shape": list(mel.shape),
        "mel_spectrogram": mel[:, ::max(1, mel.shape[1] // 200)].tolist(),  # downsampled for UI
        "rir_features": rir,
        "breathing_features": breathing,
        "feature_vector": feature_vector,
    }
