# Next Word Predictor

A clean, modular **Next Word Prediction** project implemented in Python and PyTorch.

This repository implements a simple yet extensible word-level language model that learns to predict the **next word** given a sequence of previous words, similar to the suggestion feature on smartphone keyboards.

The codebase is structured following a professional `src/` layout and is ready to be published to GitHub:

- Clear separation between **configuration**, **data pipeline**, **model**, **training loop**, and **inference**.
- Reproducible experiments (seed control, configuration objects).
- Command-line interface for training and inference.

GitHub: <https://github.com/mobinyousefi-cs>

---

## Features

- Word-level tokenization with a configurable minimum frequency threshold.
- PyTorch-based LSTM language model for next-word prediction.
- Flexible configuration via Python dataclasses.
- Training script with:
  - Mini-batch SGD with Adam
  - Learning rate scheduling (optional)
  - Gradient clipping
  - Periodic checkpointing
- Inference script with:
  - Top-*k* next-word suggestions
  - Temperature-based sampling (optional)
- Ready for extension to character-level or subword-level models.

---

## Project Structure

```text
next-word-predictor/
├── .editorconfig
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
├── src/
│   └── next_word_predictor/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── model.py
│       ├── train.py
│       ├── predict.py
│       └── utils.py
└── tests/
    └── test_basic.py
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/mobinyousefi-cs/next-word-predictor.git
cd next-word-predictor
```

> Adjust the URL once you create the GitHub repository.

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -e .
```

This will install the package in editable mode and pull the dependencies defined in `pyproject.toml`.

---

## Preparing a Dataset

This project is intentionally generic: it works with **any plain text corpus**.

1. Collect or download a text file, for example:
   - A public domain book from Project Gutenberg.
   - A cleaned collection of chat logs, news articles, etc.
2. Save the corpus to a text file, for example:

```text
./data/corpus.txt
```

3. The training script will:
   - Read the text file.
   - Build a vocabulary.
   - Split into training and validation sets.

> You can later swap this corpus with larger datasets (e.g., WikiText, news corpora) by just changing the file path.

---

## Training the Model

The main entry point for training is `next_word_predictor.train`.

### Basic training command

```bash
python -m next_word_predictor.train \
    --data-path ./data/corpus.txt \
    --output-dir ./runs/exp1
```

Key arguments:

- `--data-path`: Path to your raw text corpus.
- `--output-dir`: Directory where checkpoints and vocab files are stored.
- `--seq-len`: Sequence length (default: 30).
- `--batch-size`: Batch size (default: 64).
- `--epochs`: Number of training epochs (default: 10).
- `--embedding-dim`: Size of word embeddings (default: 128).
- `--hidden-dim`: Hidden size of the LSTM (default: 256).

Example with custom hyperparameters:

```bash
python -m next_word_predictor.train \
    --data-path ./data/corpus.txt \
    --output-dir ./runs/exp2 \
    --seq-len 20 \
    --batch-size 128 \
    --epochs 15 \
    --embedding-dim 200 \
    --hidden-dim 300
```

During training you will see loss values and, optionally, validation metrics printed to the console.

---

## Using the Trained Model for Next-Word Prediction

After training finishes, the best model checkpoint and vocabulary are saved under `--output-dir`.

Use the `predict` module to generate next word suggestions:

```bash
python -m next_word_predictor.predict \
    --checkpoint ./runs/exp1/best_model.pt \
    --vocab-path ./runs/exp1/vocab.json \
    --prompt "artificial intelligence is" \
    --top-k 5
```

Output example:

```text
Prompt: artificial intelligence is
Top-5 predictions:
1. transforming (p=0.21)
2. becoming     (p=0.18)
3. a            (p=0.15)
4. the          (p=0.09)
5. increasingly (p=0.07)
```

You can also enable temperature sampling to get more diverse predictions:

```bash
python -m next_word_predictor.predict \
    --checkpoint ./runs/exp1/best_model.pt \
    --vocab-path ./runs/exp1/vocab.json \
    --prompt "deep learning has" \
    --top-k 5 \
    --temperature 0.8
```

---

## Running Tests

Basic tests are provided under the `tests/` directory. To run them:

```bash
pip install pytest
pytest
```

---

## Extending the Project

Some ideas to extend this project for research or portfolio work:

- Replace the LSTM with:
  - GRU
  - Transformer-based encoder
- Switch from word-level to **subword-level** tokenization (e.g., Byte Pair Encoding).
- Use pre-trained word embeddings (GloVe, FastText) to initialize the embedding layer.
- Integrate Hugging Face `datasets` to automatically download standard corpora.
- Add a small web UI (e.g., Streamlit or Gradio) for interactive next-word suggestion.

---

## License

This project is released under the **MIT License**. See the `LICENSE` file for details.

