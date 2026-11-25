
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from src.rag_pipeline import rag_chain

load_dotenv()

st.set_page_config(
    page_title="PulseOps AI",
    page_icon="📦",
    layout="wide"
)

# Application Header
st.title("📦 PulseOps AI")
st.subheader("AI E-Commerce Operations Assistant")

st.write(
    "Your intelligent assistant for inventory, demand forecasting, "
    "suppliers, purchase orders, deliveries and returns."
)

st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Sidebar
with st.sidebar:
    st.header("📦 PulseOps AI")
    st.write("Advanced Intent-Based RAG Assistant")

    st.divider()

    st.subheader("Supported Features")
    st.write("📄 Document Question Answering")
    st.write("💬 Conversation History")
    st.write("🤖 General Conversation")
    st.write("🔄 Follow-up Questions")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
question = st.chat_input("Ask PulseOps anything...")

if question:

    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate AI response
    
    with st.chat_message("assistant"):
        with st.spinner("PulseOps AI is thinking..."):
            try:
                result = rag_chain.invoke({
                    "input": question,
                    "chat_history": st.session_state.chat_history[-10:]
                })

                answer = result["answer"]
                intent = result["intent"]
                sources = result.get("sources", [])

                st.markdown(answer)
                st.info(f"Detected Intent: {intent}")

                if sources:
                    st.write("📚 **Sources:**")
                    for source in sources:
                        st.write(f"- {source}")

                st.session_state.chat_history.append(
                    HumanMessage(content=question)
                )

                st.session_state.chat_history.append(
                    AIMessage(content=answer)
                )

            except Exception as e:
                answer = f"Sorry, something went wrong: {str(e)}"
                st.error(answer)
