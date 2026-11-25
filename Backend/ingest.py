from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "DATA"
VECTORSTORE_FOLDER = PROJECT_ROOT / "vectorstore"

# Find all PDF files
pdf_files = [
    file for file in DATA_FOLDER.iterdir()
    if file.is_file() and file.suffix.lower() == ".pdf"
]

if not pdf_files:
    print("No PDF files found in DATA folder!")
    exit()

print(f"Found {len(pdf_files)} PDF files.")

# Load all PDFs
documents = []

for pdf_file in pdf_files:
    loader = PyPDFLoader(str(pdf_file))
    pdf_documents = loader.load()

    documents.extend(pdf_documents)

    print(f"Loaded: {pdf_file.name} ({len(pdf_documents)} pages)")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"\nTotal PDF pages loaded: {len(documents)}")
print(f"Total text chunks created: {len(chunks)}")

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vector database
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

# Save vector database
vectorstore.save_local(str(VECTORSTORE_FOLDER))
print("\n===================================")
print("All PDFs successfully stored in FAISS!")
print("PulseOps AI knowledge base updated.")
print("===================================")