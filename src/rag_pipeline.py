
from src.config import retriever
from langchain_core.runnables import RunnableLambda

from src.chains import (
    contextualize_chain,
    document_answer_chain,
    intent_chain,
    history_answer_chain,
    general_chain
)


def retrieve_documents(inputs):
    standalone_question = contextualize_chain.invoke({
        "input": inputs["input"],
        "chat_history": inputs["chat_history"]
    })

    documents = retriever.invoke(standalone_question)

    context = "\n\n".join(
        [doc.page_content for doc in documents]
    )

    sources = sorted(set(
        doc.metadata.get("source", "Unknown")
        for doc in documents
    ))

    if not context.strip():
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
        "sources": sources
    }


def route_request(inputs):
    question = inputs["input"]
    history = inputs["chat_history"]

    intent = intent_chain.invoke({
        "input": question,
        "chat_history": history
    }).strip().upper()

    print("\nDetected Intent:", intent)

    if intent == "DOCUMENT":
        result = retrieve_documents(inputs)

    elif intent == "HISTORY":
        answer = history_answer_chain.invoke({
            "input": question,
            "chat_history": history
        })

        result = {
            "answer": answer,
            "intent": "HISTORY",
            "sources": []
        }

    else:
        answer = general_chain.invoke({
            "input": question,
            "chat_history": history
        })

        result = {
            "answer": answer,
            "intent": "GENERAL",
            "sources": []
        }

    print("\nAI Answer:", result["answer"])

    if result["sources"]:
        print("\nSources:")
        for source in result["sources"]:
            print("-", source)

    return result


rag_chain = RunnableLambda(route_request)
