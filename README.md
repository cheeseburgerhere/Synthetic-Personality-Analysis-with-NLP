# 🧠 Synthetic Personality Analysis with NLP

> Predicting hobbies and interests from synthetic personality descriptions using embedding-based NLP techniques.

**Authors:** Yasin Yeşilyurt & Abdullah Arda Gündoğdu  
**Course:** BIL 470 — Senior Project  

---

## 📌 Overview

This project explores whether **natural language descriptions of a person's personality** can predict their **hobbies and interests**. We build a full NLP pipeline that:

1. **Extracts & cleans** hobby data from a large synthetic persona dataset ([NVIDIA Nemotron Personas](https://huggingface.co/datasets/nvidia/Nemotron-Personas))
2. **Reduces hobby noise** via compression-based clustering (Normalized Compression Distance)
3. **Generates dense embeddings** using Qwen 3 language models (0.6B & 8B)
4. **Predicts hobbies** from persona embeddings using k-NN retrieval and Cross-Encoder re-ranking
5. **Learns a direct projection** from persona → hobby embedding space via a deep neural network

---

## 🏗️ Project Structure

```
.
├── 📁 embeds/                          # Pre-computed persona embeddings
│   ├── v0.1/                           # Qwen 3 0.6B persona embeddings
│   ├── v0.1_8B/                        # Qwen 3 8B persona embeddings
│   └── embed_exploration.ipynb         # Embedding analysis notebook
│
├── 📁 initial_expedition/              # Early exploration & prototyping
│   ├── dataset_expedition_hobbies.ipynb
│   ├── dataset_fixed.ipynb
│   ├── canonical_embeddings_*.npy      # V1 embedding outputs
│   └── semantically_merged_hobbies*.csv
│
├── 📁 expedition_v2/                   # V2 pipeline (production)
│   ├── pipeline_runner.py              # End-to-end orchestrator
│   ├── clustering_ncd.py               # NCD-based hobby clustering
│   ├── embedding_factory.py            # Qwen embedding generator
│   ├── evaluate_quality.py             # Cluster & embedding evaluation
│   ├── output/                         # Generated clusters + embeddings
│   └── logs/                           # Pipeline execution logs
│
├── 📁 prediction/                      # Hobby prediction engines
│   ├── base_predictor.py               # Abstract predictor interface
│   ├── prediction_MatrixKNN.py         # Matrix k-NN predictor
│   ├── prediction_CrossEncoder.py      # Two-stage Cross-Encoder predictor
│   ├── similarity_utils.py             # Vector similarity & k-NN utilities
│   ├── final_dataset.csv               # Final persona dataset
│   ├── matrixKNN.ipynb                 # Interactive MatrixKNN exploration
│   └── cross_encoding.ipynb            # Cross-Encoder experiments
│
├── 📁 vector_to_vector_reggression/    # Deep learning projection
│   ├── vector_projection_model.py      # HobbyProjector neural network
│   ├── training_util.py                # Training loop & dataset class
│   ├── y_projection.py                 # Persona → hobby dataset builder
│   ├── training_nemotron.ipynb         # Training notebook
│   └── projection_dataset.npz         # Cached (X, y) dataset
│
├── 📁 training_results/                # Saved model checkpoints (.pth)
│
├── test_prediction.py                  # Integration tests for MatrixKNN
├── test_similarity.py                  # Unit tests for similarity utils
├── extract_pdf_text.py                 # PDF text extraction utility
└── .gitignore
```

---

## 🔬 Methodology

### 1. Data Preparation

The raw data comes from the [NVIDIA Nemotron-Personas](https://huggingface.co/datasets/nvidia/Nemotron-Personas) dataset — a large-scale synthetic dataset of personality profiles. We extract hobby/interest fields and flatten them into a unique list of activity strings.

### 2. Hobby Clustering (NCD)

Raw hobbies are noisy and contain many near-duplicates (e.g., *"playing soccer"* vs. *"soccer"*). We use **Normalized Compression Distance (NCD)** — a parameter-free string similarity metric based on Kolmogorov complexity — combined with **Agglomerative Clustering** to merge semantically identical hobbies into canonical groups.

```
NCD(x, y) = [ C(xy) − min(C(x), C(y)) ] / max(C(x), C(y))
```

This dramatically reduces the hobby vocabulary while preserving semantic diversity.

### 3. Embedding Generation

We generate dense vector representations for both canonical hobbies and persona descriptions using **Qwen 3 Embedding** models:

| Model | Parameters | Use Case |
|-------|-----------|----------|
| `Qwen/Qwen3-Embedding-0.6B` | 0.6B | Lightweight experiments |
| `Qwen/Qwen3-Embedding-8B` | 8B | High-quality production embeddings |

Embeddings are produced via mean pooling over the last hidden state of tokenized inputs.

### 4. Hobby Prediction

We implement two prediction strategies, both following a common `BasePredictor` interface:

- **MatrixKNN** — Vectorized exact k-nearest neighbor search using matrix multiplication for cosine/euclidean similarity. Fast and effective for embedding retrieval.
- **CrossEncoder (Two-Stage)** — First retrieves candidates via MatrixKNN, then re-ranks them with a `cross-encoder/ms-marco-MiniLM-L-6-v2` model for improved precision.

### 5. Vector-to-Vector Projection (Deep Learning)

Instead of retrieval-based matching, we also train a **HobbyProjector** neural network to directly learn the mapping from persona embedding space → hobby embedding space:

```
Persona Vector (4096-d)  →  MLP(4096→2048→2048→4096)  →  Hobby Vector (4096-d)
```

- Architecture: 3-layer MLP with ReLU + Dropout (0.2) + L2-normalized output
- Loss: `CosineEmbeddingLoss` (maximizes directional similarity)
- Optimizer: Adam (lr=1e-4)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for embedding generation and training)

### Installation

```bash
git clone https://github.com/cheeseburgerhere/Synthetic-Personality-Analysis-with-NLP.git
cd Synthetic-Personality-Analysis-with-NLP
pip install -r requirements.txt
```

> **Note:** If a `requirements.txt` is not yet available, install the following core dependencies:
> ```bash
> pip install numpy pandas scikit-learn torch transformers sentence-transformers matplotlib
> ```

### Running the Pipeline

**1. NCD Clustering + Embedding Generation:**
```bash
python expedition_v2/pipeline_runner.py --input_path all_hobbies.json --limit 0
```
- `--limit 0` processes all hobbies; set a smaller value for testing.

**2. Build Projection Dataset:**
```bash
python vector_to_vector_reggression/y_projection.py
```

**3. Train the Projector Model:**
```bash
python vector_to_vector_reggression/training_util.py
```

**4. Evaluate Clustering Quality:**
```bash
python expedition_v2/evaluate_quality.py
```

### Running Tests

```bash
python test_similarity.py
python test_prediction.py
```

---

## 📊 Key Results

- **Hobby Clustering** significantly reduces vocabulary size while preserving semantic coverage.
- **Matrix k-NN** provides fast, reliable baseline predictions using cosine similarity.
- **Cross-Encoder re-ranking** improves prediction nuance by jointly encoding persona-hobby pairs.
- **HobbyProjector** learns a direct, generalizable mapping between embedding spaces.

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.9+ |
| **Deep Learning** | PyTorch |
| **NLP Models** | Qwen 3 Embedding (0.6B, 8B), MiniLM Cross-Encoder |
| **ML Utilities** | scikit-learn, NumPy, Pandas |
| **Transformers** | Hugging Face `transformers`, `sentence-transformers` |
| **Visualization** | Matplotlib, t-SNE |

---

## 📄 License

This project was developed as part of the BIL 470 Senior Project course. Please contact the authors for usage permissions.
