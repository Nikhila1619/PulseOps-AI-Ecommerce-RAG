
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load Hugging Face embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load saved FAISS database
vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

# Ask a question
query = input("Enter your question: ")

# Retrieve relevant documents
results = vectorstore.similarity_search(query, k=3)

print("\nRelevant Information:\n")

for i, result in enumerate(results, start=1):
    print(f"\nResult {i}:")
    print(result.page_content)
    print("Source:", result.metadata.get("source"))
