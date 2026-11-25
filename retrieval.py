
def retrieve_documents(inputs):
    # Step 1: Rewrite the question using conversation history
    standalone_question = contextualize_chain.invoke({
        "input": inputs["input"],
        "chat_history": inputs["chat_history"]
    })

    print("\nOriginal Question:", inputs["input"])
    print("Rewritten Question:", standalone_question)

    # Step 2: Retrieve relevant documents
    documents = retriever.invoke(standalone_question)

    # Step 3: Extract document context
    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    # Step 4: Extract source filenames
    sources = sorted(set(
        doc.metadata.get("source", "Unknown")
        for doc in documents
    ))

    # Step 5: Generate answer
    if not documents or not context.strip():
        answer = "I don't know based on the provided documents."
    else:
        answer = document_answer_chain.invoke({
            "input": standalone_question,
            "context": context,
            "chat_history": inputs["chat_history"]
        })

    return {
        "answer": answer,
        "intent": "DOCUMENT",
        "sources": sources,
        "rewritten_question": standalone_question
    }
