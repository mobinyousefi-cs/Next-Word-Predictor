#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Next Word Predictor
File: test_basic.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=

Description:
Basic unit tests to sanity-check the data pipeline and model shapes.

Usage:
pytest tests/test_basic.py

Notes:
- These tests are minimal but ensure that core components are wired correctly.

============================================================================
"""

from __future__ import annotations

from pathlib import Path

import torch

from next_word_predictor.data import Vocab, NextWordDataset
from next_word_predictor.model import NextWordLM


def test_vocab_and_dataset_shapes(tmp_path: Path) -> None:
    text = "deep learning is fun when learning is deep"
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(text, encoding="utf-8")

    tokens = text.split()
    vocab = Vocab.build(tokens, min_freq=1)
    token_ids = vocab.encode(tokens)

    ds = NextWordDataset(token_ids, seq_len=3, pad_id=vocab.pad_id)
    assert len(ds) == len(tokens) - 3

    x, y = ds[0]
    assert x.shape == (3,)
    assert isinstance(y.item(), int)


def test_model_forward() -> None:
    vocab_size = 20
    model = NextWordLM(vocab_size=vocab_size, embedding_dim=8, hidden_dim=16)

    x = torch.randint(0, vocab_size, (4, 5))  # (batch, seq_len)
    logits, hidden = model(x)

    assert logits.shape == (4, 5, vocab_size)
    h, c = hidden
    assert h.shape[0] == model.lstm.num_layers
