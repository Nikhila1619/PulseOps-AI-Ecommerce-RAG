import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS database
vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

# Create prompt
prompt = ChatPromptTemplate.from_template("""
You are PulseOps AI, an E-commerce Operations Assistant.

Answer the question using only the following context.

If the answer is not available in the context, say:
"I could not find this information in the company documents."

Context:
{context}

Question:
{question}

Answer:
""")

print("\nWelcome to PulseOps AI!")
print("Ask questions about company policies and operations.")
print("Type 'exit' to stop.\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("Thank you for using PulseOps AI!")
        break

    # Retrieve relevant documents
    docs = vectorstore.similarity_search(question, k=3)

    context = "\n\n".join(doc.page_content for doc in docs)

    # Generate answer
    messages = prompt.format_messages(
        context=context,
        question=question
    )

    response = llm.invoke(messages)

    print("\nPulseOps AI:", response.content)
    print("\n" + "-" * 60)