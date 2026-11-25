from langchain_core.output_parsers import StrOutputParser
from src.config import llm
from src.prompts import  (
    intent_prompt,
    contextualize_prompt,
    answers_prompt, # Based on Document
    history_prompt,
    general_prompt

)

# Intent classfication chain
intent_chain=(
    intent_prompt
    |llm
    |StrOutputParser()
    )

# rewritting Prompt Chain
contextualize_chain=(
    contextualize_prompt
    |llm
    |StrOutputParser()

)

# Document nswer chain
document_answer_chain=(
    answers_prompt
    |llm
    |StrOutputParser()
)

# History answer chain
history_answer_chain = (
    history_prompt
    |llm
    |StrOutputParser()
)

# General Conversation Chain
general_chain=(
    general_prompt
    |llm
    |StrOutputParser()
)