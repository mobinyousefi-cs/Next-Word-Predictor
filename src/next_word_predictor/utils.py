#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Next Word Predictor
File: utils.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=

Description:
Utility functions for reproducibility, device management, and checkpoint I/O.

Usage:
from next_word_predictor.utils import set_seed, get_device

Notes:
- Central place for small helpers to keep other modules clean.

============================================================================
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str | None = None) -> torch.device:
    """Return a torch.device, preferring CUDA if available.

    Parameters
    ----------
    preferred:
        If "cuda" or "cpu", force that device when available.
    """

    if preferred is not None:
        if preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(
    checkpoint_path: Path,
    model_state: Dict[str, Any],
    optimizer_state: Dict[str, Any],
    vocab: Dict[str, Any],
    training_config: Dict[str, Any],
) -> None:
    """Save model, optimizer, vocabulary, and config to a checkpoint.

    The checkpoint is split into a binary PyTorch file for tensors and a JSON
    file for metadata to keep things readable.
    """

    checkpoint_path = checkpoint_path.with_suffix(".pt")
    meta_path = checkpoint_path.with_suffix(".meta.json")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state": model_state,
        "optimizer_state": optimizer_state,
    }, checkpoint_path)

    metadata = {
        "vocab": vocab,
        "training_config": training_config,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_checkpoint(checkpoint_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load a checkpoint and return model state, optimizer state, vocab, config."""

    checkpoint_path = checkpoint_path.with_suffix(".pt")
    meta_path = checkpoint_path.with_suffix(".meta.json")

    data = torch.load(checkpoint_path, map_location="cpu")
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    return (
        data["model_state"],
        data["optimizer_state"],
        meta["vocab"],
        meta["training_config"],
    )
