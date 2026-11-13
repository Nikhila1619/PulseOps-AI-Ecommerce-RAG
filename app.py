
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="PulseOps AI",
    page_icon="📦",
    layout="wide"
)

st.title("📦 PulseOps AI")
st.subheader("AI E-Commerce Operations Assistant")

st.write(
    "Ask questions about inventory, demand forecasting, "
    "suppliers, purchase orders, deliveries and returns."
)


@st.cache_resource
def load_components():
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is missing from your .env file.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

    return vectorstore, llm


prompt = ChatPromptTemplate.from_template("""
You are PulseOps AI, an E-Commerce Operations Assistant.

Answer the question using only the following context.

If the answer is not available in the context, say:
"I could not find this information in the company documents."

Context:
{context}

Question:
{question}

Answer:
""")


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("📦 PulseOps AI")
    st.write("Your E-Commerce Operations Assistant")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


try:
    vectorstore, llm = load_components()
except Exception as e:
    st.error(f"Unable to load PulseOps: {e}")
    st.stop()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Ask PulseOps anything...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching company documents..."):

            docs = vectorstore.similarity_search(question, k=3)

            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            messages = prompt.format_messages(
                context=context,
                question=question
            )

            response = llm.invoke(messages)
            answer = response.content

            st.markdown(answer)

            sources = sorted(set(
                doc.metadata.get("source", "Unknown")
                for doc in docs
            ))

            with st.expander("View Sources"):
                for source in sources:
                    st.write(source)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
