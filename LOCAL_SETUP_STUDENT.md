# Running ANLP Notebooks on Your Own Machine

> **Recommended:** Use the course JupyterHub at
> <https://www.anlp-course-um.nl> — everything is pre-installed and ready
> to go. The instructions below are **only** for students who prefer to
> work locally (e.g. for faster GPU access, offline work, or when the
> cluster is busy).

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone the Repository](#2-clone-the-repository)
3. [Create a Python Environment](#3-create-a-python-environment)
4. [Install PyTorch with CUDA](#4-install-pytorch-with-cuda)
5. [Install Python Dependencies](#5-install-python-dependencies)
6. [Download NLTK Data](#6-download-nltk-data)
7. [Download spaCy Models](#7-download-spacy-models)
8. [Verify GPU Access](#8-verify-gpu-access)
9. [OpenAI API Key (Assignments 11, 12 & 16)](#9-openai-api-key-assignments-11-12--16)
10. [Assignment-specific Notes](#10-assignment-specific-notes)
11. [Submitting Your Work](#11-submitting-your-work)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.13 (recommended and tested). Python 3.10–3.12 should also work. Python 3.14+ may cause compatibility issues with some packages. |
| **pip** | Latest version (`python -m pip install --upgrade pip`) |
| **Git** | To clone the repository |
| **GPU** | **NVIDIA GPU required** for all deep learning assignments (06-16). See [Section 4](#4-install-pytorch-with-cuda) for requirements. |
| **OS** | Windows 10/11, macOS 12+, or Linux (Ubuntu 20.04+) |
| **RAM** | Minimum 8 GB; 16 GB recommended |
| **Disk space** | ~10 GB free (for packages, models, and datasets) |

---

## 2. Clone the Repository

```bash
git clone https://github.com/TextMiningUM/ANLP-Student.git
cd ANLP-Student
```

> **Note:** If you're reading this locally, you've likely already cloned the repository.

---

## 3. Create a Python Environment

We strongly recommend using a **virtual environment** to avoid conflicts with
other projects.

### Option A — conda (recommended if you have Anaconda/Miniconda)

```bash
conda create -n anlp python=3.13 -y
conda activate anlp
```

### Option B — venv (built-in)

```bash
python -m venv .venv

# Activate on Linux/macOS:
source .venv/bin/activate

# Activate on Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# Activate on Windows (cmd):
.\.venv\Scripts\activate.bat
```

---

## 4. Install PyTorch with CUDA

**All ANLP assignments require PyTorch with CUDA support.** Install this FIRST, before the other dependencies.

### GPU Requirements

| Component | Requirement |
|---|---|
| **GPU** | NVIDIA GPU with Compute Capability 5.0+ |
| **Supported GPUs** | GTX 1660+, RTX 20/30/40/50 series, A100, H100, L40, RTX 6000, etc. |
| **NVIDIA Drivers** | Version 525.60.13 or newer (check with `nvidia-smi`) |
| **VRAM** | Minimum 4 GB; 8+ GB recommended for assignments 07-12 |

### Installation

Install PyTorch 2.6.0 with CUDA 12.4 support:

```bash
# Works on Windows, Linux, and macOS (with NVIDIA GPU)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

> **Important:** This exact version (2.6.0+cu124) is tested and used throughout the course.
> CUDA 12.4 is backward compatible with all modern NVIDIA drivers (525.60.13+).

---

## 5. Install Python Dependencies

A `requirements.txt` is provided in the repository root:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs **all** remaining packages needed across all 19 assignments. The full
install takes roughly 5–15 minutes depending on your network and hardware.

> **Note:** PyTorch (torch/torchvision) is NOT in requirements.txt because it must be
> installed separately with CUDA support (see Section 4 above).

---

## 6. Download NLTK Data

Several assignments rely on NLTK corpora and models. Run this **once** after
installing the Python packages:

```python
import nltk
nltk.download([
    'punkt',
    'punkt_tab',
    'stopwords',
    'wordnet',
    'omw-1.4',
    'words',
    'movie_reviews',
    'brown',
    'universal_tagset',
    'book',
    'tagsets_json',
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng',
    'treebank',
    'vader_lexicon',
    'maxent_ne_chunker',
    'maxent_ne_chunker_tab',
])
```

Or from the command line:

```bash
python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4 words movie_reviews brown universal_tagset book tagsets_json averaged_perceptron_tagger averaged_perceptron_tagger_eng treebank vader_lexicon maxent_ne_chunker maxent_ne_chunker_tab
```

---

## 7. Download spaCy Models

Several assignments require spaCy English models:

```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

---

## 8. Verify GPU Access

A test script `test_gpu.py` is provided in the repository root. Run it:

```bash
python test_gpu.py
```

Expected output (with GPU):
```
PyTorch version: 2.6.0+cu124
CUDA available: True
CUDA version: 12.4
Number of GPUs: 1

GPU 0: NVIDIA GeForce RTX 4070 Ti
  VRAM: 12.0 GB
  Compute capability: 8.9

✅ GPU support is working correctly!
   You can now run the deep learning assignments with GPU acceleration.
```

### Troubleshooting GPU Issues

**Problem:** `CUDA available: False`

**Solutions:**
1. Check if you have an NVIDIA GPU: `nvidia-smi`
2. Update NVIDIA drivers: <https://www.nvidia.com/Download/index.aspx>
3. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchvision torchaudio -y
   pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
   ```
4. Verify you're using the correct Python environment

**Problem:** `OutOfMemoryError` during training

**Solutions:**
1. **Reduce batch size:** Change `batch_size=16` to `batch_size=4` or lower
2. **Use gradient accumulation:**
   ```python
   # Instead of batch_size=32, use batch_size=8 with 4 accumulation steps
   trainer = Trainer(..., per_device_train_batch_size=8, gradient_accumulation_steps=4)
   ```
3. **Enable gradient checkpointing:**
   ```python
   model.gradient_checkpointing_enable()
   ```
4. **Use mixed precision:**
   ```python
   trainer = Trainer(..., fp16=True)  # or bf16=True for newer GPUs
   ```
5. **Use model quantization (8-bit or 4-bit):**
   ```python
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained("gpt2", load_in_8bit=True)
   ```

**Problem:** Training is slow even with GPU

**Check:**
1. Verify GPU is actually being used:
   ```python
   print(next(model.parameters()).device)  # Should show 'cuda:0'
   ```
2. Monitor GPU utilization: `nvidia-smi -l 1` (refreshes every second)
3. Ensure data is on GPU: `inputs = inputs.to('cuda')`

---

## 9. OpenAI API Key (Assignments 11, 12 & 16)

Assignments 11 (*Fine-tuning LLMs*), 12 (*Multi-modal Models*), and 16 
(*Agents*) use the **OpenAI API**. You will need:

1. An OpenAI account with **billing enabled** at <https://platform.openai.com>.
2. An API key generated at <https://platform.openai.com/api-keys>.

The cost is modest — expect roughly **€2–€5** for completing these assignments
with `gpt-4o-mini` or `gpt-4o`.

The notebooks will prompt you for the key using `getpass` (it is never stored
in the notebook). Alternatively, set it as an environment variable:

```bash
# Linux/macOS
export OPENAI_API_KEY="sk-..."

# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
```

> **Important:** Never commit your API key to Git. The `.gitignore` should
> already exclude sensitive files, but double-check before pushing.

---

## 10. Assignment-specific Notes

### Assignment 01 — Tokenization
- Fetches live webpages via `urllib.request`; requires internet access.
- Uses various tokenization libraries (NLTK, spaCy, etc.).

### Assignment 04 — Syntax and Semantics
- Requires parsing libraries for syntax tree visualization.
- Uses `svgling` for tree rendering.

### Assignment 05 — Statistical NLP
- Covers n-gram models and language model evaluation.
- Downloads text corpora from NLTK.

### Assignment 06 — PyTorch
- Introduction to PyTorch tensor operations.
- GPU highly recommended for training exercises.

### Assignment 07 — Transformers
- Introduction to the Transformers library.
- Downloads pre-trained models from HuggingFace (~500MB each).

### Assignment 08 — Encoder Models
- Fine-tuning BERT and similar models.
- GPU strongly recommended (CPU training may take 20+ minutes).

### Assignment 09 — Decoder Models
- Working with GPT-style models.
- Text generation tasks; GPU recommended.

### Assignment 10 — XAI in NLP
- Explainability methods for NLP models.
- Uses attention visualizations and LIME/SHAP.

### Assignment 11 — Fine-tuning LLMs
- Fine-tuning large language models.
- **GPU with 8GB+ VRAM required** for most exercises.
- OpenAI API needed for comparison tasks.

### Assignment 12 — Multi-modal Models
- Vision-language models (CLIP, BLIP, etc.).
- Downloads large model checkpoints (~1–2GB).
- GPU recommended; OpenAI API for GPT-4V tasks.

### Assignment 13 — Speech Recognition
- Uses speech processing libraries (librosa, soundfile).
- May require audio file downloads.

### Assignment 14 — Abstracting and MT
- Machine translation and text summarization.
- Uses sequence-to-sequence models.

### Assignment 15 — Dialog
- Dialog systems and conversational AI.
- Covers both rule-based and neural approaches.

### Assignment 16 — Agents
- AI agents and autonomous systems.
- Uses OpenAI Agents SDK; requires `openai>=1.40.0`.

### Major Assignments

#### A1 — NLP Fundamentals
- Comprehensive assessment of basic NLP concepts (Assignments 01-05).

#### A2 — Deep Learning for NLP
- Applied deep learning for NLP (Assignments 06-10).

#### A3 — NLP Applications
- Advanced NLP applications (Assignments 11-16).

---

## 11. Submitting Your Work

Even if you develop locally, you should **submit via the workflow in the repository**:

1. **Before submitting:** Restart the kernel and run all cells to ensure
   the notebook executes cleanly from top to bottom:
   - In Jupyter: `Kernel → Restart & Run All`
   - In VS Code: Click the restart button, then run all cells
2. Copy your completed notebook to the `Submitted Work/<topic>/` folder:
   ```bash
   cp "Personal Workspace/01 tokenization/01_ANLP_Tokenization_2026_2027.ipynb" \
      "Submitted Work/01 tokenization/"
   ```
3. **Keep the original filename** — do not rename the notebook.
4. If using JupyterHub: Push your changes or manually upload the notebook.
5. If working locally: Sync your `Submitted Work/` folder to the submission system.

> **Important:** Test your submission by opening the notebook in `Submitted Work/`
> and running it one more time to ensure everything works.

---

## 12. Troubleshooting

### "ModuleNotFoundError: No module named '...'"
You likely missed a dependency. Make sure you installed from the provided
`requirements.txt` and that your virtual environment is activated.

### CUDA / GPU not detected
- Verify you installed the CUDA version of PyTorch (not the CPU version).
- Check that your NVIDIA drivers are up-to-date: `nvidia-smi`.
- Ensure `torch.cuda.is_available()` returns `True`.

### NLTK data not found
Re-run the NLTK download commands in [Section 5](#5-download-nltk-data).
You can also manually set the NLTK data path:
```python
import nltk
nltk.data.path.append('/path/to/your/nltk_data')
```

### HuggingFace model download is slow
Models are cached in `~/.cache/huggingface/`. The first load takes time;
subsequent loads are instant. If your network is restricted, consider
downloading models on a different network and copying the cache folder.

### Package version conflicts
If you encounter version incompatibilities, try creating a fresh environment.
The course is tested with **Python 3.13**, but 3.10–3.12 should also work:

```bash
# With conda
conda create -n anlp-fresh python=3.13 -y
conda activate anlp-fresh
pip install -r requirements.txt

# With venv
python3.13 -m venv .venv-fresh
source .venv-fresh/bin/activate  # or .venv-fresh\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

### "RuntimeError: CUDA out of memory"
Reduce the batch size in training cells, or switch to CPU by setting:
```python
device = torch.device("cpu")
```

### spaCy model not found
Re-run the spaCy model download commands:
```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

---

## Summary of External Services & Costs

| Service | Assignments | Cost | Required? |
|---|---|---|---|
| Course JupyterHub | All | Free | Primary platform |
| OpenAI API | 11, 12, 16 | ~€2–€5 | Yes, for these assignments |
| Internet access | Most | — | Yes (data & model downloads) |

---

*Last updated: July 2026*
