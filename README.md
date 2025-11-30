# EchoHybrid RAG System

![EchoHybrid Logo](https://img.shields.io/badge/EchoHybrid-v3.0-blue) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

EchoHybrid is a secure, research-grade **Retrieval-Augmented Generation (RAG)** system that implements a **two-layer architecture** for privacy-preserving document retrieval:

- **First Layer** → Document ingestion, chunking, redaction, embedding, and secure vector storage
- **Second Layer** → Multi-vector hybrid retrieval with echo-based refinement and chunk hydration

## 🌟 Key Features

### 🔒 Privacy-First Architecture

- **Embeddings-Only Storage**: Raw text never leaves your infrastructure
- **Local Text Hydration**: Full text content retrieved from secure local storage
- **Optional PII Redaction**: Built-in sensitive data handling

### 🚀 Advanced Retrieval

- **Multi-Vector Search**: Combines dense, sparse, and temporal vectors
- **Echo Refinement**: Dynamic query optimization without LLM dependencies
- **Confidence-Based Filtering**: Smart result ranking and filtering
- **Negative Memory**: Prevents repeated low-quality results

### ⚡ Performance Optimized

- **Parallel Processing**: Async operations for maximum throughput
- **Efficient Chunking**: Intelligent document segmentation
- **Configurable Pipeline**: Tune for your specific use case
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

````bash
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
### 🚀 Quick Start

#### 1. Prerequisites
- Python 3.8+
- Docker (for Qdrant)
- Git

#### 2. Setup Environment
```bash
# Clone repository
git clone https://github.com/yourusername/echo-rag.git
cd echo-rag

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Qdrant (Docker)
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
````

#### 3. Initialize the System

```bash
# Reset Qdrant collection
python -m EchoHybrid.reset_collection
```

#### 4. Ingest Documents

```bash
# Process a single file
python -m EchoHybrid.ingestion.runner "path/to/your/document.pdf" --recreate-collection

# Or process multiple files
python -m EchoHybrid.ingestion.runner "path/to/documents/" --pattern "*.pdf"
```

This will create:

- `parsed_output/`: Raw parsed documents
- `chunking_output/`: Processed chunks with metadata
- `generated_embeddings/`: Local backup of embeddings

And store vectors in Qdrant collection: `document_embeddings`

✅ Step 4: Verify Secure Storage
bash
Copy code
python check_embeddings_only.py
python check_vectors.py
Expected:

✅ Collection exists

✅ Total embeddings stored

✅ No raw text in Qdrant

#### 5. Run Queries

```bash
python -m EchoHybrid.retrieval.engine "Your search query here"
```

Example Query:

```bash
python -m EchoHybrid.retrieval.engine "What are the main types of distribution graphs and their primary uses?"
```

Example Output:

```json
[
  {
    "rank": 1,
    "score": 0.923,
    "text": "The main types of distribution graphs include histograms, box plots, and violin plots. Histograms are best for showing the shape of continuous data, while box plots excel at comparing distributions across categories. Violin plots combine the benefits of both, showing the full distribution with a box plot overlay.",
    "source_file": "M2 S3- Distribution Display.pdf",
    "page_or_time": "12",
    "highlight_terms": ["distribution", "graphs", "histograms", "box plots"],
    "found_in_round": "echo_1",
    "modality": "text"
  }
]
```

## 🏗️ Project Structure

```
EchoHybrid/
│
├── ingestion/                # First Layer: Document Processing
│   ├── runner.py            # Main ingestion pipeline
│   ├── parser.py            # Document parsing (PDF, DOCX, TXT)
│   ├── chunker.py           # Intelligent text chunking
│   ├── embedder.py          # Text embedding generation
│   ├── vector_store.py      # Qdrant vector database interface
│   └── redactor.py          # PII and sensitive data handling
│
├── retrieval/               # Second Layer: Retrieval Engine
│   ├── engine.py            # Main retrieval pipeline
│   ├── triple_vector_search.py  # Multi-vector search
│   ├── echo_discovery.py    # Query refinement
│   ├── echo_parallel_search.py  # Parallel search execution
│   ├── rrf_fusion.py        # Result fusion and ranking
│   ├── echo_parent.py       # Document context handling
│   ├── negative_memory.py   # Low-quality result filtering
│   ├── recipe_selector.py   # Search strategy selection
│   └── confidence_gate.py   # Result confidence scoring
│
├── chunking_output/         # Processed document chunks (JSON)
├── parsed_output/           # Raw parsed documents
├── generated_embeddings/    # Local embedding backups
│
├── config.py               # Configuration settings
├── reset_collection.py     # Qdrant collection management
├── check_vectors.py        # Vector storage verification
└── requirements.txt        # Python dependencies
```

## ⚙️ Configuration

### Ingestion Settings

| Argument                | Description                  | Default                         |
| ----------------------- | ---------------------------- | ------------------------------- |
| `input_path`            | File or directory to process | Required                        |
| `--output-dir`          | Directory for chunk output   | `chunking_output`               |
| `--no-redact`           | Disable PII redaction        | `False`                         |
| `--chunk-size`          | Maximum chunk size in tokens | `1000`                          |
| `--chunk-overlap`       | Token overlap between chunks | `200`                           |
| `--pattern`             | File pattern for directories | `*`                             |
| `--embed-model`         | SentenceTransformer model    | `all-MiniLM-L6-v2`              |
| `--device`              | Processing device            | `cuda` if available, else `cpu` |
| `--qdrant-collection`   | Qdrant collection name       | `document_embeddings`           |
| `--recreate-collection` | Recreate Qdrant collection   | `False`                         |

### Retrieval Settings (in `config.py`)

```python
# Qdrant Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "document_embeddings"

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Must match selected model

# Search Parameters
TOP_K_RESULTS = 10
SIMILARITY_THRESHOLD = 0.15
```

## 🔧 Supported Models

### Pre-trained Models

- `all-MiniLM-L6-v2` (384d) - **Recommended** - Fast and accurate for most use cases
- `all-mpnet-base-v2` (768d) - Higher accuracy, larger size
- `multi-qa-MiniLM-L6-cos-v1` - Optimized for question answering

### Custom Models

Any model from the [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html) library can be used by specifying the model name in the configuration.

To use a custom model:

1. Ensure it's compatible with SentenceTransformers
2. Update `EMBEDDING_MODEL` in `config.py`
3. Set `EMBEDDING_DIM` to match the model's output dimension

## 🔒 Security & Privacy

### Data Protection

- **No Raw Text in Database**: Only numerical embeddings are stored in Qdrant
- **Local Text Storage**: Original documents and chunks remain on your infrastructure
- **Secure Processing**: In-memory processing with automatic cleanup

### PII Handling

- **Redaction Module**: Built-in support for detecting and redacting:
  - Email addresses
  - Phone numbers
  - Credit card numbers
  - Social security numbers
  - Custom patterns via regex

### Compliance

- **GDPR/CCPA Ready**: Designed with privacy regulations in mind
- **Audit Trail**: Full control over data processing and storage
- **No External Dependencies**: All processing happens locally

## 🐛 Troubleshooting

### Common Issues

#### Qdrant Connection Issues

```bash
# Check if Qdrant is running
curl http://localhost:6333

# If not, start Qdrant
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

#### Reset Collection

```bash
# Clear all data and start fresh
python -m EchoHybrid.reset_collection
```

#### GPU Verification

```bash
# Check GPU availability
nvidia-smi

# Check PyTorch GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## � Performance Tips

1. **Hardware Acceleration**

   - Use CUDA-enabled GPU for faster embeddings
   - Allocate at least 8GB RAM for Qdrant

2. **Chunking Strategy**

   - For technical documents: 500-1000 tokens
   - For general text: 1000-2000 tokens
   - Adjust overlap to 10-20% of chunk size

3. **Qdrant Optimization**

   - Monitor with Qdrant dashboard at `http://localhost:6333/dashboard`
   - Adjust `hnsw_ef` and `hnsw_m` parameters for large collections

4. **Batch Processing**
   - Process documents in batches of 10-50
   - Use `--device cuda` for GPU acceleration

🤝 Contributing
Fork the repo

Create a branch

Commit changes

Add tests

Open a PR

📄 License
MIT License

🔄 Version History

## 📜 Version History

### v3.0.0 (Current)

- Added echo-based query refinement
- Implemented multi-vector hybrid search
- Added chunk hydration system
- Enhanced confidence scoring
- Improved error handling and logging

### v2.0.0

- Integrated retrieval pipeline
- Added negative memory system
- Implemented RRF fusion
- Added confidence gating

### v1.2.0

- Enhanced chunking strategies
- Added verification tools
- Improved error handling

### v1.1.0

- Embeddings-only storage
- Secure vector database integration
- Redaction module

### v1.0.0

- Initial release
- Basic document processing
- Vector storage and retrieval

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For support or questions, please open an issue in the repository.
