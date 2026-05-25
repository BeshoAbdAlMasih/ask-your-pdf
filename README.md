# 📄 Ask Your PDF

A local RAG (Retrieval-Augmented Generation) app that lets you chat with any PDF using LangChain, FAISS, HuggingFace embeddings, and a locally running Ollama LLM — no API keys, no cloud, fully private.

---

## Demo

> Upload a PDF → Ask a question → Get an answer grounded in the document.

---

## Features

- 📥 Upload any PDF and extract its text automatically
- 🔍 Semantic search over document chunks via FAISS vector store
- 🤗 HuggingFace sentence embeddings (`all-mpnet-base-v2`) running locally
- 🦙 Local LLM inference via Ollama (default: `llama3`)
- ⚡ Cached embeddings — rebuilds only when the document changes
- 🔒 100% local — no data leaves your machine

---

## Tech Stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| PDF parsing | PyPDF2 |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | HuggingFace `sentence-transformers/all-mpnet-base-v2` |
| Vector store | FAISS (CPU) |
| LLM | Ollama (`llama3`) |
| Chain | LangChain retrieval chain |

---

## Prerequisites

- Python **3.11** (required — 3.12+ may cause dependency conflicts)
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- [Ollama](https://ollama.com) installed and running
- CUDA-capable GPU (optional, for faster embeddings)

---

## Installation

**1. Clone the repo**
```bash
git clone https://github.com/BeshoAbdAlMasih/ask-your-pdf.git
cd ask-your-pdf
```

**2. Create and activate a virtual environment (Python 3.11)**
```bash
uv venv .venv --python 3.11
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS
```

**3. Install dependencies**
```bash
uv pip install -r requirements.txt
```

**4. (Optional) Enable GPU for embeddings**

Install PyTorch with CUDA support:
```bash
uv pip uninstall torch
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then set `model_kwargs={'device': 'cuda'}` in `app.py`. Skip this step to run on CPU.

**5. Pull the LLM model via Ollama**
```bash
ollama pull llama3
```

**6. Create a `.env` file** (optional, for future API integrations)
```bash
touch .env   # Windows: type nul > .env
```

---

## Usage

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser, upload a PDF, and start asking questions.

---

## Project Structure

```
ask-your-pdf/
├── app.py               # Main Streamlit application
├── requirements.txt
├── .env                 # Environment variables (not committed)
├── .gitignore
└── README.md
```

---

## Requirements

```
streamlit
python-dotenv
PyPDF2
langchain==0.3.25
langchain-community==0.3.24
langchain-core==0.3.59
langchain-huggingface
langchain-text-splitters
faiss-cpu
sentence-transformers
```

> **GPU note:** `faiss-cpu` is used because `faiss-gpu` has no wheels for Python 3.11+. Embedding inference still runs on GPU via PyTorch independently.

---

## How It Works

```
PDF Upload
    │
    ▼
Text Extraction (PyPDF2)
    │
    ▼
Chunking (RecursiveCharacterTextSplitter, 1000 chars / 200 overlap)
    │
    ▼
Embedding (HuggingFace all-mpnet-base-v2)  ──── cached with st.cache_resource
    │
    ▼
FAISS Vector Store (CPU)
    │
    ▼
User Question ──► Retriever ──► Relevant Chunks
                                      │
                                      ▼
                              Ollama LLM (llama3)
                                      │
                                      ▼
                                   Answer
```

---

## Configuration

| Parameter | Location | Default |
|---|---|---|
| LLM model | `app.py` → `Ollama(model=...)` | `llama3` |
| Embedding model | `app.py` → `HuggingFaceEmbeddings(model_name=...)` | `all-mpnet-base-v2` |
| Chunk size | `app.py` → `chunk_size` | `1000` |
| Chunk overlap | `app.py` → `chunk_overlap` | `200` |
| Device | `app.py` → `model_kwargs['device']` | `cpu` (`cuda` if GPU enabled) |

---

## License

MIT