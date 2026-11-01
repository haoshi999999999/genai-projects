from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2
    )

def ask_llm(llm, prompt: str):
    return llm.invoke([HumanMessage(content=prompt)])
