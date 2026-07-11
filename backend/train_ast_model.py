"""
AcousticSpace - Fine-tune AST for real/fake audio classification

Downloads the pretrained MIT/ast-finetuned-audioset-10-10-0.4593 checkpoint
from HuggingFace, freezes most of the transformer backbone (for feasible
CPU fine-tuning), and trains the classification head + last two transformer
blocks on labeled audio clips.

⚠️ REQUIRES INTERNET ACCESS to huggingface.co (to download the base
checkpoint). Run this on your own machine, not inside a network-restricted
sandbox.

Usage:
    python train_ast_model.py                # synthetic demo data (fast)
    python train_ast_model.py --epochs 5

To train on real ASVspoof data instead of the synthetic demo set, replace
`load_training_samples()` with a loader that returns a list of
(waveform: np.ndarray @ 16kHz, label: int) tuples where 0=real, 1=fake --
nothing else in this script needs to change.
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    ASTForAudioClassification, ASTFeatureExtractor,
    Trainer, TrainingArguments,
)

from model_ast import CHECKPOINT, FINE_TUNED_DIR, LABEL2ID, ID2LABEL, SAMPLE_RATE
from synthetic_audio import generate_dataset


def load_training_samples(n_samples=240):
    """Swap this out for a real ASVspoof loader when you have dataset
    access -- must return [(waveform_np_float32_16khz, label_int), ...]."""
    return generate_dataset(n_samples=n_samples)


class ASTDataset(Dataset):
    """Pre-extracts AST input features for every sample up front (dataset
    is small enough to fit comfortably in memory for the demo)."""

    def __init__(self, samples, extractor: ASTFeatureExtractor):
        self.features = []
        self.labels = []
        for waveform, label in samples:
            feats = extractor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            self.features.append(feats["input_values"].squeeze(0))
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_values": self.features[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def freeze_backbone_except_last_layers(model: ASTForAudioClassification, n_unfrozen=2):
    """Freezes everything except the classifier head and the last
    `n_unfrozen` transformer encoder blocks -- keeps CPU fine-tuning
    tractable while still meaningfully adapting the model."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    encoder_layers = model.audio_spectrogram_transformer.layers
    for layer in encoder_layers[-n_unfrozen:]:
        for p in layer.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--n-samples", type=int, default=240)
    args = parser.parse_args()

    print(f"Downloading base checkpoint '{CHECKPOINT}' from HuggingFace...")
    extractor = ASTFeatureExtractor.from_pretrained(CHECKPOINT)
    model = ASTForAudioClassification.from_pretrained(
        CHECKPOINT,
        num_labels=2,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
    )
    freeze_backbone_except_last_layers(model, n_unfrozen=2)

    print(f"Generating {args.n_samples} synthetic training clips...")
    samples = load_training_samples(n_samples=args.n_samples)
    split = int(len(samples) * 0.85)
    train_ds = ASTDataset(samples[:split], extractor)
    eval_ds = ASTDataset(samples[split:], extractor)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = float((preds == labels).mean())
        return {"accuracy": acc}

    training_args = TrainingArguments(
        output_dir="./ast_train_ckpt",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=5,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)

    os.makedirs(FINE_TUNED_DIR, exist_ok=True)
    model.save_pretrained(FINE_TUNED_DIR)
    extractor.save_pretrained(FINE_TUNED_DIR)
    print(f"Saved fine-tuned AST model to {FINE_TUNED_DIR}")


if __name__ == "__main__":
    main()
