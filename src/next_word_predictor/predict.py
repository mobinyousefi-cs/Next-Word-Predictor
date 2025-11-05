#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Next Word Predictor
File: predict.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=

Description:
Inference utilities and CLI for generating next-word predictions from a
trained model checkpoint.

Usage:
python -m next_word_predictor.predict --checkpoint ./runs/exp1/best_model.pt --vocab-path ./runs/exp1/vocab.json --prompt "deep learning is"

Notes:
- Supports top-k predictions and temperature-based sampling.

============================================================================
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import torch
import typer

from .data import Vocab, load_vocab, tokenize
from .model import NextWordLM
from .utils import get_device, load_checkpoint


app = typer.Typer(add_completion=False)


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("Temperature must be > 0.")
    return logits / temperature


def _top_k_predictions(
    probs: torch.Tensor,
    k: int,
    vocab: Vocab,
) -> List[Tuple[str, float]]:
    k = min(k, probs.numel())
    values, indices = torch.topk(probs, k)
    results: List[Tuple[str, float]] = []
    for prob, idx in zip(values.tolist(), indices.tolist()):
        token = vocab.itos[idx]
        results.append((token, float(prob)))
    return results


def load_model_for_inference(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[NextWordLM, Vocab]:
    """Load model and vocabulary for inference."""

    model_state, _opt_state, vocab_data, training_cfg = load_checkpoint(checkpoint_path)

    vocab = Vocab.from_json(vocab_data)

    model = NextWordLM(
        vocab_size=vocab.size,
        embedding_dim=training_cfg.get("embedding_dim", 128),
        hidden_dim=training_cfg.get("hidden_dim", 256),
        num_layers=training_cfg.get("num_layers", 2),
        dropout=training_cfg.get("dropout", 0.2),
    )
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    return model, vocab


def predict_next_word(
    prompt: str,
    checkpoint: Path,
    device_str: str | None = None,
    top_k: int = 5,
    temperature: float = 1.0,
) -> List[Tuple[str, float]]:
    """High-level API for next-word prediction.

    Returns a list of (token, probability) pairs sorted by probability.
    """

    device = get_device(device_str)
    model, vocab = load_model_for_inference(checkpoint, device)

    tokens = tokenize(prompt)
    if not tokens:
        raise ValueError("Prompt must contain at least one token.")

    token_ids = vocab.encode(tokens)
    x = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model.predict_next(x)
        logits = _apply_temperature(logits, temperature)
        probs = _softmax(logits).squeeze(0)

    return _top_k_predictions(probs, top_k, vocab)


@app.command()
def main(
    checkpoint: Path = typer.Option(..., exists=True, help="Path to model checkpoint (.pt)."),
    vocab_path: Path = typer.Option(..., exists=True, help="Path to vocab JSON file."),
    prompt: str = typer.Option(..., help="Prompt text."),
    top_k: int = typer.Option(5, help="Number of predictions to show."),
    temperature: float = typer.Option(1.0, help="Sampling temperature (>0)."),
    device_str: str | None = typer.Option(None, help="Preferred device: 'cpu' or 'cuda'."),
) -> None:
    """Command-line interface for next-word prediction."""

    device = get_device(device_str)
    vocab = load_vocab(vocab_path)

    # Load model using metadata from checkpoint
    model_state, _opt_state, vocab_data, training_cfg = load_checkpoint(checkpoint)
    model = NextWordLM(
        vocab_size=vocab.size,
        embedding_dim=training_cfg.get("embedding_dim", 128),
        hidden_dim=training_cfg.get("hidden_dim", 256),
        num_layers=training_cfg.get("num_layers", 2),
        dropout=training_cfg.get("dropout", 0.2),
    )
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    tokens = tokenize(prompt)
    if not tokens:
        typer.echo("Prompt must contain at least one token.")
        raise typer.Exit(code=1)

    token_ids = vocab.encode(tokens)
    x = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model.predict_next(x)
        logits = _apply_temperature(logits, temperature)
        probs = _softmax(logits).squeeze(0)

    results = _top_k_predictions(probs, top_k, vocab)

    typer.echo(f"Prompt: {prompt}")
    typer.echo("Top predictions:")
    for i, (token, prob) in enumerate(results, start=1):
        typer.echo(f"{i:2d}. {token:20s} (p={prob:.4f})")


if __name__ == "__main__":  # pragma: no cover
    app()
