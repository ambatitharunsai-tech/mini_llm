# Hema AI – Experimental NLP and Language Model Workspace

Hema AI is an experimental assistant project combining a Python NLP/modeling codebase with a TypeScript/Express web server. The repository includes chatbot and retrieval utilities, training scripts, dataset preparation code, model checkpoints/logs and a browser-facing application.

> **Project status:** this repository contains multiple development paths (a Python Flask/model implementation and a TypeScript Express application). They are separate entry points in the current source tree; do not assume the Express server automatically loads the PyTorch model.

## Capabilities and modules

- **NLP utilities:** text normalization, tokenization, intent handling, entity extraction and query matching.
- **Transformer experiment:** a GPT-style decoder-only PyTorch model with positional encoding and autoregressive generation helpers.
- **Retrieval experiments:** vector-search utilities and stored vector-answer metadata.
- **Web assistant:** Express server with a local Q&A set, conversation context, feedback/learning endpoints and static asset serving.
- **Python application:** Flask-based assistant code with optional document, image and web-search integrations.
- **Training workflow:** dataset preparation, training scripts, training logs and an external-GPU notebook/script.

## Technology

| Area | Tools |
|---|---|
| Python model and NLP | Python, PyTorch, NumPy, pandas, scikit-learn |
| Retrieval | sentence-transformers, FAISS |
| Python web service | Flask, Requests |
| TypeScript web service | Node.js, TypeScript, Express, tsx |
| Document and image support | pypdf, Pillow |
| Data and experiments | CSV datasets, notebooks, model/log artifacts |

## Repository layout

```text
.
├── data/                  # Dataset preparation and training data
├── models/                # Training logs and vector/retrieval artifacts
├── scripts/               # External-GPU training helpers
├── src/
│   ├── app.py             # Python Flask assistant entry point
│   ├── model.py           # PyTorch transformer model
│   ├── train.py           # Training code
│   ├── generate.py        # Text generation utilities
│   ├── bpe_tokenizer.py   # Tokenizer implementation
│   ├── nlp_engine.py      # NLP and reasoning helpers
│   └── vector_db.py       # Vector retrieval utilities
├── server.ts              # TypeScript/Express web application
├── train_external_gpu.ipynb
├── package.json
└── requirements.txt
```

## Run the TypeScript web application

Requirements: a recent Node.js release and npm.

```bash
npm install
npm run dev
```

The Express application listens on port `3000` by default. Set `PORT` to change it.

## Set up the Python environment

Use Python 3.10+ and a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate      # Windows PowerShell
pip install -r requirements.txt
```

The Python application entry point is `src/app.py`. Start it from the repository root with:

```bash
python src/app.py
```

Some optional features require additional external services, model artifacts, browser binaries, or API credentials. Install/configure only the integrations you intend to use.

## Model training

The repository includes scripts and a notebook for dataset preparation and training. Before starting a run:

1. Review `data/prepare_dataset.py` and `data/generate_full_dataset.py`.
2. Check the dataset format and the tokenizer/model vocabulary alignment.
3. Review the hyperparameters and checkpoint paths in `src/train.py`.
4. Use a GPU runtime with enough VRAM for the chosen sequence length, batch size and model configuration.
5. Preserve training logs and checkpoint metadata so experiments can be reproduced.

Training code is experimental. A successful training step or a lower training loss does not, by itself, demonstrate that the assistant gives accurate or useful answers. Evaluate generation on held-out prompts before using a checkpoint.

## Development checks

```bash
npm run build
```

The package build script runs the TypeScript compiler in no-emit mode. Python tests or evaluations should be run using the scripts and test files supplied with the specific experiment.

## Security and data

- Keep API tokens and service credentials outside source control.
- Do not put private user conversations or personal data into public training datasets without appropriate rights and consent.
- Treat scraped or third-party content as untrusted and follow the source site's terms.
- Do not expose debugging/context endpoints on a public production deployment without access controls.

## License

No license file is currently listed in the repository root. Contact the repository owner before reusing this project's code or datasets.
