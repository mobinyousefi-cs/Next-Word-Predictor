#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Next Word Predictor
File: config.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=

Description:
Configuration dataclasses for model architecture and training hyperparameters.

Usage:
Import the configuration objects and either use the defaults or customize
fields programmatically.

Notes:
- Designed to be lightweight and framework-agnostic.
- Can be later replaced by a YAML/TOML-based config system if needed.

============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class ModelConfig:
    """Configuration for the language model architecture."""

    vocab_size: int
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.2


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for the training process."""

    data_path: Path
    output_dir: Path
    seq_len: int = 30
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    val_split: float = 0.1
    min_freq: int = 2
    seed: int = 42
    device: Optional[str] = None  # "cuda", "cpu", or None for auto

    def to_dict(self) -> dict:
        """Return a serializable dictionary representation of the config."""

        data = asdict(self)
        data["data_path"] = str(self.data_path)
        data["output_dir"] = str(self.output_dir)
        return data
