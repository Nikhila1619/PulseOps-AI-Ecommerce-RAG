
# PulseOps – AI E-Commerce Operations Assistant

## Project Overview
PulseOps is a Retrieval-Augmented Generation (RAG) based chatbot designed to answer questions related to e-commerce operations.

It retrieves relevant information from PDF documents and uses a Large Language Model (LLM) to generate accurate, context-based responses.

## Objectives
- Provide answers about inventory management.
- Understand demand forecasting.
- Retrieve supplier policies.
- Explain purchase order procedures.
- Answer delivery-related questions.
- Provide return and cancellation information.

## Technologies Used
- Python
- LangChain
- Hugging Face Embeddings
- FAISS Vector Database
- Groq LLM
- Streamlit
- PyPDF

## Project Architecture
1. Load PDF documents.
2. Split documents into smaller text chunks.
3. Generate embeddings using Hugging Face.
4. Store embeddings in FAISS.
5. Retrieve relevant documents based on user questions.
6. Generate answers using the Groq LLM.
7. Display answers through the Streamlit interface.

## Project Structure
- DATA/ – PDF documents
- ingest.py – Document loading and vector database creation
- retrieval.py – Similarity search
- chatbot.py – RAG chatbot logic
- app.py – Streamlit user interface
- requirements.txt – Required Python libraries
- .env – API key configuration (not uploaded)

## Installation

Clone this repository:

    git clone https://github.com/Nikhila1619/PulseOps-AI-Ecommerce-RAG

Install dependencies:

    pip install -r requirements.txt

Create a .env file in the project folder and add:

    GROQ_API_KEY=your_groq_api_key

## Run the Project

First, create the FAISS vector database:

    python ingest.py

Then launch the chatbot:

    streamlit run app.py

## Example Questions
- What is the reorder point?
- What is demand forecasting?
- How are suppliers evaluated?
- What is the purchase order process?
- What is the delivery process?
- What is the return policy?

## Note
The PDF policy documents included in this educational project are sample documents created for demonstration purposes. They do not represent live company policies or operational data.

## Author
AI Data Engineering Internship Project

