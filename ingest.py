
import glob
import os

from dotenv import load_dotenv
from supabase import create_client

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore

# Step 1: Load environment variables
load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("Supabase credentials are missing from .env")

# Step 2: Find PDF documents
pdf_files = glob.glob("DATA/*.pdf")

if not pdf_files:
    raise FileNotFoundError("No PDF files found inside the DATA folder.")

# Step 3: Load PDF documents
documents = []

for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    pdf_documents = loader.load()
    documents.extend(pdf_documents)
    print(f"Loaded: {os.path.basename(pdf_file)}")

print("\nTotal PDF files:", len(pdf_files))
print("Total pages:", len(documents))

# Step 4: Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))

# Step 5: Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 6: Connect to Supabase
supabase_client = create_client(
    supabase_url,
    supabase_key
)

# Step 7: Store document chunks and embeddings in Supabase
vectorstore = SupabaseVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    client=supabase_client,
    table_name="documents",
    query_name="match_documents"
)

print("\nSupabase Vector Database Created Successfully!")
print("Uploaded chunks:", len(chunks))
