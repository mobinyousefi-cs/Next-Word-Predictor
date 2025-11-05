#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Next Word Predictor
File: __init__.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=

Description:
Package initialization for the Next Word Predictor project. Exposes high-level
APIs for training and next-word prediction.

Usage:
python -m next_word_predictor.train [options]
python -m next_word_predictor.predict [options]

Notes:
- This file defines the package version and convenient imports.

============================================================================
"""

from __future__ import annotations

from .predict import predict_next_word

__all__ = ["predict_next_word"]

__version__ = "0.1.0"
