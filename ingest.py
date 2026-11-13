
import glob
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Find all PDF documents
pdf_files = glob.glob("data/*.pdf")

if not pdf_files:
    raise FileNotFoundError("No PDF files found inside the data folder.")

# Step 2: Load all PDF documents
documents = []

for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    pdf_documents = loader.load()
    documents.extend(pdf_documents)
    print(f"Loaded: {os.path.basename(pdf_file)}")

print("\nTotal PDF files:", len(pdf_files))
print("Total pages:", len(documents))

# Step 3: Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))

# Step 4: Create Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 5: Create FAISS vector database
vectorstore = FAISS.from_documents(chunks, embeddings)

# Step 6: Save the vector database
vectorstore.save_local("vectorstore")

print("\nFAISS Vector Database Created Successfully!")
