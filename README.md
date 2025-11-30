# EchoHybrid RAG System

EchoHybrid is a secure, research-grade **Retrieval-Augmented Generation (RAG)** system that implements a **two-layer architecture**:

- **First Layer** → Document ingestion, chunking, redaction, embedding, and secure vector storage
- **Second Layer** → Deterministic hybrid retrieval with echo-based refinement

This system is designed for **privacy-preserving RAG**, where **only embeddings are stored in the vector database**, and all raw text is kept locally.

---

## 🚀 Features

### ✅ First Layer – Secure Ingestion

- Multi-format Document Parsing (PDF, DOCX, TXT, etc.)
- Hybrid Intelligent Chunking with configurable size and overlap
- Optional Sensitive Information Redaction
- Embedding Generation using Sentence Transformers
- Secure Embeddings-Only Storage in Qdrant (no raw text)
- Batch Processing for multiple documents

### ✅ Second Layer – Deterministic Retrieval (Integrated)

- Dense Vector Search via Qdrant
- Sparse Keyword Search from local `chunking_output`
- Echo-based Query Refinement (without LLM)
- Reciprocal Rank Fusion (RRF)
- Secure Text Hydration from local storage
- Fully Offline, LLM-Free Retrieval Engine
- CLI-based Retrieval Interface

---

## 📋 Prerequisites

- Python **3.9 – 3.13**
- Qdrant Server (local)
- Docker (recommended for Qdrant)
- CUDA-capable GPU (optional, for faster embeddings)

---

## 🛠 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd EchoHybrid
2. Create Virtual Environment
bash
Copy code
python -m venv venv
venv\Scripts\activate   # Windows
3. Install Dependencies
bash
Copy code
pip install -r requirements.txt
4. Environment Variables
Create a .env file in the project root:

env
Copy code
QDRANT_HOST=localhost
QDRANT_PORT=6333
🚀 Quick Start Guide
✅ Step 1: Start Qdrant (Docker Recommended)
bash
Copy code
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
Dashboard:

text
Copy code
http://localhost:6333/dashboard
✅ Step 2: Reset the Vector Collection
bash
Copy code
python -m reset_collection
✅ Step 3: Ingest Documents (First Layer)
Single File
bash
Copy code
python -m ingestion.runner "path/to/your/document.pdf" --recreate-collection
Multiple Files
bash
Copy code
python -m ingestion.runner "path/to/documents/" --pattern "*.pdf"
This will generate:

parsed_output/

chunking_output/

generated_embeddings/

And store embeddings in:

Qdrant collection: document_embeddings

✅ Step 4: Verify Secure Storage
bash
Copy code
python check_embeddings_only.py
python check_vectors.py
Expected:

✅ Collection exists

✅ Total embeddings stored

✅ No raw text in Qdrant

🔎 Second Layer: Retrieval Engine (Integrated)
Location:

text
Copy code
EchoHybrid/retrieval/engine.py
✅ Run a Query
bash
Copy code
python -m retrieval.engine "What are the main types of distribution graphs and their primary uses?"
✅ Example Output
json
Copy code
[
  {
    "rank": 1,
    "score": 0.153,
    "text": "…retrieved document text…",
    "source_file": "M2 S3- Distribution Display.pdf",
    "page_or_time": "",
    "highlight_terms": [],
    "found_in_round": "echo_1",
    "modality": "text"
  }
]
📁 Project Structure
text
Copy code
EchoHybrid/
├── ingestion/
│   ├── runner.py
│   ├── parser.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   └── redactor.py
├── retrieval/
│   ├── engine.py
│   ├── recipe_selector.py
│   └── (other retrieval modules)
├── generated_embeddings/
├── chunking_output/
├── parsed_output/
├── reset_collection.py
├── check_vectors.py
├── check_embeddings_only.py
├── config.py
├── app.py
└── requirements.txt
⚙️ Ingestion Configuration (CLI)
Argument	Description	Default
input_path	File or directory	Required
--output-dir	Chunk output	chunking_output
--no-redact	Disable redaction	Enabled
--chunk-size	Max chunk size	1000
--chunk-overlap	Token overlap	200
--pattern	File pattern	*
--embed-model	Embedding model	all-MiniLM-L6-v2
--device	Processing device	cpu/cuda
--qdrant-collection	Qdrant collection	document_embeddings
--recreate-collection	Recreate	False

🤖 Supported Models
all-MiniLM-L6-v2 (384d)

all-mpnet-base-v2 (768d)

multi-qa-MiniLM-L6-cos-v1

Custom SentenceTransformer models

🔒 Security Features
✅ Embeddings-only storage in Qdrant

✅ No raw text in the vector DB

✅ Optional PII redaction

✅ Local-only text hydration

🐛 Troubleshooting
Qdrant Not Running
bash
Copy code
curl http://localhost:6333
Clear Collection
bash
Copy code
python -m reset_collection
Check GPU
bash
Copy code
nvidia-smi
📊 Performance Tips
Use GPU if available

Adjust chunk size

Avoid extremely large PDFs

Monitor Qdrant dashboard

🤝 Contributing
Fork the repo

Create a branch

Commit changes

Add tests

Open a PR

📄 License
MIT License

🔄 Version History
v1.0.0 – Secure ingestion + embeddings

v1.1.0 – Embeddings-only Qdrant storage

v1.2.0 – Chunking + verification

v2.0.0 – Integrated deterministic hybrid retrieval (EchoHybrid)

```
