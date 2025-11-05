#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Next Word Predictor
File: model.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=

Description:
PyTorch implementation of a simple LSTM-based language model used for
next-word prediction.

Usage:
from next_word_predictor.model import NextWordLM

Notes:
- The model predicts the distribution over the vocabulary for the next word
  given a sequence of token IDs.

============================================================================
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class NextWordLM(nn.Module):
    """A basic LSTM language model for next-word prediction."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape (batch_size, seq_len) with token IDs.
        hidden:
            Optional initial hidden state for the LSTM.

        Returns
        -------
        logits:
            Tensor of shape (batch_size, seq_len, vocab_size).
        hidden:
            Final hidden state tuple.
        """

        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, hidden

    def predict_next(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Return logits for only the last time step in the sequence."""

        logits, hidden = self.forward(x, hidden)
        last_logits = logits[:, -1, :]
        return last_logits, hidden
