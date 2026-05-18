# 🤖 AI-Generated Text Detector — NLP Classification Pipeline

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?style=flat&logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat)](https://huggingface.co)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-95.73%25-brightgreen?style=flat)]()

> Can a machine tell if text was written by a human or an AI? Turns out — yes. And the gap between "counting words" and "reading context" is enormous.

---

## 📌 Overview

This project builds an end-to-end NLP pipeline to classify text as **human-written** or **AI-generated**, comparing three model architectures from statistical baselines to fine-tuned transformers.

**Key result:** Fine-tuned DistilBERT achieves **95.5% test accuracy** — a 10.5 percentage point improvement over the TF-IDF baseline.

---

## 🏗️ Architecture

```
Dataset (20,000+ samples)
    ├── Public dataset (Nature Scientific Reports)
    └── Custom hybrid dataset (Wikipedia passages + LLM rewrites)
            ↓
    Pair-Aware Splitting (prevents data leakage)
            ↓
    ┌─────────────────────────────────┐
    │  Model Tier 1: TF-IDF + LR     │ → 85.0% accuracy
    │  Model Tier 2: TF-IDF + SVC    │ → 85.4% accuracy  
    │  Model Tier 3: DistilBERT      │ → 95.5% accuracy ✅
    └─────────────────────────────────┘
```

---

## 📊 Results

| Model | Features | Val Accuracy | Test Accuracy | F1 Score |
|---|---|---|---|---|
| Logistic Regression (Unigram) | TF-IDF Unigram | 85.00% | 85.96% | 0.86 |
| Logistic Regression (Bigram) | TF-IDF Bigram | 86.15% | 85.96% | 0.86 |
| LinearSVC | TF-IDF Bigram | 85.39% | — | — |
| DistilBERT (lr=2e-5) | Contextual embeddings | 93.20% | — | — |
| **DistilBERT (lr=5e-5)** | **Contextual embeddings** | **95.53%** | **95.73%** | **0.96** |

---

## 🔑 Key Technical Decisions

### 1. Custom Hybrid Dataset
Built a matched dataset of **3,868 samples** (1,934 human Wikipedia passages + 1,934 AI rewrites on identical topics), combined with a public dataset of 19,866 cleaned samples — **23,734 total**. Forces the model to learn writing *style*, not subject matter.

**Top machine-indicating words confirmed:** `overall, summary, typically, provides, topic, sentence, allows`
**Top human-indicating words confirmed:** `just, really, actually, basically, probably, thing`

### 2. Pair-Aware Splitting
Standard random splits would leak information — matched pairs could end up across train/test partitions, inflating results. Implemented pair-aware splitting so matched pairs always stay in the same partition.

### 3. Two Learning Rate Experiments (DistilBERT)
- `lr = 2e-5` — conservative, stable convergence
- `lr = 5e-5` — faster convergence, marginally higher variance

---

## 🛠️ Tech Stack

- **Python 3.10**
- **PyTorch** — model training
- **HuggingFace Transformers** — DistilBERT fine-tuning
- **Scikit-Learn** — TF-IDF, Logistic Regression, LinearSVC
- **Pandas / NumPy** — data processing
- **Matplotlib / Seaborn** — visualisation

---

## 📁 Repository Structure

```
ai-text-detector/
  └── AI_Text_Detector.ipynb      # Full pipeline notebook
  └── Data.txt                    # Dataset sources and links    
  └── README.md                   #Compete Sketch on pipeliine
```

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/gayatri-kannan/ai-text-detector.git
cd ai-text-detector

# Install dependencies
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn

# Run the notebook
jupyter notebook notebook/AI_Text_Detector.ipynb
```

---

## 📚 Dataset

- **Primary:** Human vs Machine Text dataset — [Nature Scientific Reports](https://www.nature.com/articles/s41598-024-55686-2)
- **Custom:** Wikipedia passages (human) + LLM rewrites (machine) — constructed for this project

---

## 💡 Key Insight

> *The jump from 86% → 95.73% accuracy is the entire argument for why transformers changed NLP. TF-IDF counts words in isolation. DistilBERT reads the whole sentence and understands how ideas connect. The learning rate choice (2e-5 vs 5e-5) alone moved validation accuracy from 93.2% → 95.53%.*

---

## 👩‍💻 Author

**Gayatri Kannan Padma**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/gayatri-kannan-ai)
