"""
AcousticSpace - Synthetic demo audio generator

Builds toy speech-like signals with either:
  - natural reverb + irregular breathing gaps  -> labeled "real"
  - dry/minimal reverb + uniform breathing gaps -> labeled "fake"

Stands in for a real dataset (ASVspoof) so the training pipeline is fully
runnable without network/licensing access to that dataset. Swap
`generate_dataset()` for an ASVspoof loader when you have access -- nothing
else in the pipeline needs to change, since downstream code only expects
(waveform: np.ndarray, label: int) pairs at 16kHz.
"""

import numpy as np

SAMPLE_RATE = 16000
RNG = np.random.default_rng(42)


def _synthesize_voice_like(duration=3.0, sr=SAMPLE_RATE, reverb_amount=0.0,
                            breath_regular=False, n_phrases=4):
    """Toy speech-like signal: formant-ish tones with amplitude envelopes,
    separated by breath gaps, optionally convolved with a synthetic room
    impulse response."""
    total_samples = int(duration * sr)
    y = np.zeros(total_samples)
    phrase_len = total_samples // n_phrases

    pos = 0
    for _ in range(n_phrases):
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


def generate_dataset(n_samples=240):
    """Returns a list of (waveform, label) pairs. label: 0=real, 1=fake."""
    samples = []
    for i in range(n_samples):
        is_fake = i % 2 == 0
        if is_fake:
            y = _synthesize_voice_like(reverb_amount=RNG.choice([0.0, 0.05]), breath_regular=True)
        else:
            y = _synthesize_voice_like(reverb_amount=RNG.uniform(0.3, 0.7), breath_regular=False)
        samples.append((y, int(is_fake)))
    return samples
