#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Next Word Predictor
File: data.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=

Description:
Dataset and vocabulary utilities for preparing text data for next-word
prediction. Builds a word-level vocabulary and PyTorch Dataset objects that
produce (input_sequence, next_word) pairs.

Usage:
from next_word_predictor.data import build_vocab, create_dataloaders

Notes:
- Tokenization is intentionally simple (whitespace-based) to keep the
  implementation easy to follow. You can swap this for a better tokenizer.

============================================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass(slots=True)
class Vocab:
    """Simple word-level vocabulary with special tokens."""

    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def build(
        cls,
        tokens: Iterable[str],
        min_freq: int = 2,
    ) -> "Vocab":
        from collections import Counter

        counter = Counter(tokens)

        # Reserve IDs for special tokens
        itos: List[str] = [PAD_TOKEN, UNK_TOKEN]
        stoi: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}

        for word, freq in counter.items():
            if freq < min_freq:
                continue
            if word in stoi:
                continue
            stoi[word] = len(itos)
            itos.append(word)

        return cls(stoi=stoi, itos=itos)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.stoi.get(tok, self.stoi[UNK_TOKEN]) for tok in tokens]

    def decode(self, ids: Sequence[int]) -> List[str]:
        return [self.itos[idx] for idx in ids]

    def to_json(self) -> Dict[str, object]:
        return {"stoi": self.stoi, "itos": self.itos}

    @classmethod
    def from_json(cls, data: Dict[str, object]) -> "Vocab":
        return cls(stoi=data["stoi"], itos=data["itos"])

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    @property
    def size(self) -> int:
        return len(self.itos)


class NextWordDataset(Dataset):
    """Dataset of (input_sequence, target_word) pairs for next-word prediction."""

    def __init__(self, token_ids: List[int], seq_len: int, pad_id: int) -> None:
        super().__init__()
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.pad_id = pad_id

    def __len__(self) -> int:  # type: ignore[override]
        # Number of possible windows of size seq_len + 1
        return max(0, len(self.token_ids) - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        start = idx
        end = idx + self.seq_len
        x = self.token_ids[start:end]
        y = self.token_ids[end]
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


def tokenize(text: str) -> List[str]:
    """Very simple whitespace tokenizer.

    For a real system you would likely integrate a more robust tokenizer.
    """

    return text.strip().split()


def load_corpus(path: Path) -> List[str]:
    """Load a text file and return a list of tokens."""

    with path.open("r", encoding="utf-8") as f:
        text = f.read()
    return tokenize(text)


def create_vocab_and_datasets(
    data_path: Path,
    seq_len: int,
    min_freq: int,
    val_split: float,
) -> Tuple[Vocab, NextWordDataset, NextWordDataset]:
    """Create vocabulary and train/validation datasets from a raw corpus file."""

    tokens = load_corpus(data_path)
    vocab = Vocab.build(tokens, min_freq=min_freq)
    token_ids = vocab.encode(tokens)

    dataset = NextWordDataset(token_ids, seq_len=seq_len, pad_id=vocab.pad_id)

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check your corpus or seq_len.")

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    return vocab, train_ds, val_ds


def create_dataloaders(
    train_ds: NextWordDataset,
    val_ds: NextWordDataset,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch dataloaders for train and validation datasets."""

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def save_vocab(vocab: Vocab, path: Path) -> None:
    """Save vocabulary as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(vocab.to_json(), f, ensure_ascii=False, indent=2)


def load_vocab(path: Path) -> Vocab:
    """Load vocabulary from JSON."""

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Vocab.from_json(data)
