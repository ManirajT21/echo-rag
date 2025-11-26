EchoHybrid RAG System
A comprehensive document processing pipeline for building Retrieval-Augmented Generation (RAG) systems. This system processes various document formats, generates embeddings, and stores them in a vector database for efficient similarity search.

🚀 Features
Multi-format Document Parsing: Supports PDF, DOCX, TXT, and more

Intelligent Chunking: Hybrid chunking strategy with configurable size and overlap

Sensitive Information Redaction: Optional redaction of sensitive content

Embedding Generation: Uses Sentence Transformers for vector embeddings

Vector Storage: Stores embeddings in Qdrant vector database

Embeddings-Only Storage: Secure storage without text content in database

Batch Processing: Efficient processing of multiple documents

📋 Prerequisites
Python 3.8+

Qdrant Server (running locally or remotely)

CUDA-capable GPU (optional, for faster embeddings)

🛠 Installation
Clone the repository:

bash
git clone <repository-url>
cd echohybrid
Create virtual environment:

bash
python -m venv myvenv
source myvenv/bin/activate  # On Windows: myvenv\Scripts\activate
Install dependencies:

bash
pip install -r requirements.txt
Set up environment variables:
Create a .env file in the project root:

env
QDRANT_HOST=localhost
QDRANT_PORT=6333
🚀 Quick Start
1. Start Qdrant Server
Using Docker:

bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
Using Qdrant Binary:
Download from Qdrant GitHub and run:

bash
./qdrant
2. Reset Collection (Clean Start)
bash
python reset_embeddings.py
3. Process Documents
Single File:

bash
python -m ingestion.runner "path/to/your/document.pdf"
Directory of Files:

bash
python -m ingestion.runner "path/to/documents/" --pattern "*.pdf"
4. Verify Storage
bash
python check_embeddings_only.py
📁 Project Structure
text
echohybrid/
├── ingestion/
│   ├── runner.py          # Main pipeline runner
│   ├── parser.py          # Document parsing
│   ├── chunker.py         # Text chunking
│   ├── embedder.py        # Embedding generation
│   ├── vector_store.py    # Qdrant integration
│   └── redactor.py        # Sensitive data redaction
├── generated_embeddings/  # Local embedding storage
├── chunking_output/       # Processed chunks storage
├── parsed_output/         # Parsed documents storage
├── reset_embeddings.py    # Collection reset utility
├── check_vectors.py       # Vector storage verification
└── check_embeddings_only.py # Embeddings-only verification
⚙️ Configuration
Command Line Arguments
Argument	Description	Default
input_path	File or directory to process	Required
--output-dir	Output directory for chunks	chunking_output
--no-redact	Disable sensitive data redaction	Enabled
--chunk-size	Maximum chunk size in tokens	1000
--chunk-overlap	Token overlap between chunks	200
--pattern	File pattern for directories	*
--embed-model	Embedding model name	all-MiniLM-L6-v2
--device	Processing device	cuda/cpu
--qdrant-collection	Qdrant collection name	document_embeddings
--recreate-collection	Recreate collection	False
Supported Models
all-MiniLM-L6-v2 (Default, 384 dimensions)

all-mpnet-base-v2 (768 dimensions)

multi-qa-MiniLM-L6-cos-v1 (384 dimensions)

Custom Sentence Transformer models

🔧 Advanced Usage
Custom Processing Pipeline
python
from ingestion.runner import DocumentProcessor

# Initialize processor
processor = DocumentProcessor(
    output_dir="custom_output",
    max_chunk_size=800,
    chunk_overlap=100,
    embed_model="all-mpnet-base-v2",
    qdrant_collection="my_collection",
    recreate_collection=True
)

# Process files
chunks = processor.process_file("document.pdf")

# Verify storage
processor.verify_embeddings_storage()
Batch Processing Multiple Files
python
from ingestion.runner import DocumentProcessor

processor = DocumentProcessor()
results = processor.process_directory("/path/to/documents")
print(f"Processed {len(results)} chunks from all documents")
Custom Embedding Storage
python
from ingestion.embedder import DocumentEmbedder
from ingestion.vector_store import QdrantVectorStore

# Custom vector store
vector_store = QdrantVectorStore(
    collection_name="custom_embeddings",
    vector_size=768,
    recreate_collection=True
)

# Custom embedder
embedder = DocumentEmbedder(
    model_name="all-mpnet-base-v2",
    vector_store=vector_store,
    device="cuda"
)

# Generate and store embeddings
documents = [{"text": "Sample text", "metadata": {"source": "test"}}]
results = embedder.embed_documents(documents)
🔍 Verification Tools
Check Stored Vectors
bash
python check_vectors.py
Verify Embeddings-Only Storage
bash
python check_embeddings_only.py
Check Collection Info via curl
bash
curl "http://localhost:6333/collections/document_embeddings"
🗂 Output Files
Parsed Documents: parsed_output/{filename}_{timestamp}.json

Chunked Content: chunking_output/{filename}_chunks_{timestamp}.json

Embeddings: generated_embeddings/{source}_{model}.json

Vector Storage: Qdrant collection document_embeddings

🔒 Security Features
Text Content Isolation: Only embeddings stored in database, no raw text

Sensitive Data Redaction: Optional redaction of PII and sensitive information

Secure Metadata: Metadata stored without text content in vector database

🐛 Troubleshooting
Common Issues
Qdrant Connection Error:

Ensure Qdrant server is running: curl http://localhost:6333/

Check environment variables in .env file

Model Loading Issues:

Check internet connection for model downloads

Verify CUDA availability for GPU processing

Duplicate Vectors:

Use --recreate-collection to start fresh

Run python reset_embeddings.py to clear existing data

Memory Issues:

Reduce batch size in embedder configuration

Use smaller embedding models

Process documents in smaller batches

Debug Mode
Enable verbose logging by modifying the runner:

python
processor = DocumentProcessor(verbose=True)
📊 Performance Tips
Use GPU (--device cuda) for faster embedding generation

Adjust --chunk-size based on your document content

Use batch processing for multiple documents

Monitor Qdrant memory usage for large collections

🤝 Contributing
Fork the repository

Create a feature branch

Make your changes

Add tests

Submit a pull request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🆘 Support
For issues and questions:

Check the troubleshooting section

Review existing GitHub issues

Create a new issue with detailed description

🔄 Version History
v1.0.0: Initial release with basic RAG pipeline

v1.1.0: Added embeddings-only storage for security

v1.2.0: Enhanced chunking strategies and verification tools

Note: This system is designed for secure document processing with privacy-focused embedding storage. No raw text content is stored in the vector database, only numerical embeddings.