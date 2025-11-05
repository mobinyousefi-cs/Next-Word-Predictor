#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Next Word Predictor
File: train.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=

Description:
Command-line training script for the next-word prediction model. Handles data
loading, model initialization, training loop, validation, and checkpointing.

Usage:
python -m next_word_predictor.train --data-path ./data/corpus.txt --output-dir ./runs/exp1

Notes:
- Uses Typer for a simple, typed CLI.

============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn, optim
from tqdm import tqdm
import typer

from .config import ModelConfig, TrainingConfig
from .data import create_dataloaders, create_vocab_and_datasets, save_vocab
from .model import NextWordLM
from .utils import get_device, save_checkpoint, set_seed


app = typer.Typer(add_completion=False)


def train_one_epoch(
    model: nn.Module,
    data_loader,
    criterion,
    optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for x, y in tqdm(data_loader, desc="Train", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, _ = model(x)
        # We only care about the last time step for each sequence
        logits_last = logits[:, -1, :]

        loss = criterion(logits_last, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * x.size(0)

    return running_loss / len(data_loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader,
    criterion,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0

    for x, y in tqdm(data_loader, desc="Val", leave=False):
        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x)
        logits_last = logits[:, -1, :]
        loss = criterion(logits_last, y)
        running_loss += loss.item() * x.size(0)

    return running_loss / len(data_loader.dataset)


@app.command()
def main(
    data_path: Path = typer.Option(..., exists=True, help="Path to raw text corpus."),
    output_dir: Path = typer.Option(Path("./runs/exp1"), help="Directory for checkpoints."),
    seq_len: int = typer.Option(30, help="Sequence length for inputs."),
    batch_size: int = typer.Option(64, help="Batch size."),
    epochs: int = typer.Option(10, help="Number of training epochs."),
    lr: float = typer.Option(1e-3, help="Learning rate."),
    min_freq: int = typer.Option(2, help="Minimum token frequency for vocab."),
    hidden_dim: int = typer.Option(256, help="Hidden size of LSTM."),
    embedding_dim: int = typer.Option(128, help="Word embedding dimension."),
    num_layers: int = typer.Option(2, help="Number of LSTM layers."),
    dropout: float = typer.Option(0.2, help="Dropout rate."),
    seed: int = typer.Option(42, help="Random seed."),
    device_str: Optional[str] = typer.Option(
        None,
        help="Preferred device: 'cpu' or 'cuda'. Default: auto-detect.",
    ),
) -> None:
    """Train the next-word prediction model on a plain text corpus."""

    output_dir.mkdir(parents=True, exist_ok=True)

    training_config = TrainingConfig(
        data_path=data_path,
        output_dir=output_dir,
        seq_len=seq_len,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        min_freq=min_freq,
        seed=seed,
    )

    set_seed(training_config.seed)
    device = get_device(device_str)
    typer.echo(f"Using device: {device}")

    typer.echo("Building vocabulary and datasets...")
    vocab, train_ds, val_ds = create_vocab_and_datasets(
        data_path=training_config.data_path,
        seq_len=training_config.seq_len,
        min_freq=training_config.min_freq,
        val_split=training_config.val_split,
    )

    save_vocab(vocab, output_dir / "vocab.json")
    typer.echo(f"Vocab size: {vocab.size}")

    train_loader, val_loader = create_dataloaders(
        train_ds=train_ds,
        val_ds=val_ds,
        batch_size=training_config.batch_size,
    )

    model_config = ModelConfig(
        vocab_size=vocab.size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    model = NextWordLM(
        vocab_size=model_config.vocab_size,
        embedding_dim=model_config.embedding_dim,
        hidden_dim=model_config.hidden_dim,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_config.lr)

    best_val_loss = float("inf")
    best_ckpt_path = output_dir / "best_model.pt"

    for epoch in range(1, training_config.epochs + 1):
        typer.echo(f"Epoch {epoch}/{training_config.epochs}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        typer.echo(f"  Train loss: {train_loss:.4f}")
        typer.echo(f"  Val   loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            typer.echo("  New best model, saving checkpoint...")
            save_checkpoint(
                checkpoint_path=best_ckpt_path,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                vocab={"stoi": vocab.stoi, "itos": vocab.itos},
                training_config=training_config.to_dict(),
            )

    typer.echo(f"Training finished. Best val loss: {best_val_loss:.4f}")
    typer.echo(f"Best checkpoint saved to: {best_ckpt_path.with_suffix('.pt')}")


if __name__ == "__main__":  # pragma: no cover
    app()
