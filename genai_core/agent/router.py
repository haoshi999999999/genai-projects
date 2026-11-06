import json
import inspect

SYSTEM_PROMPT = """
You are an AI agent with tools.

Available tools:

* add_numbers(a, b)
* search_web(query)
* get_time()
* retrieve_documents(query, k)

When a tool is needed, respond with a JSON tool decision.

If no tool is needed:
{"tool": null, "answer": null}
"""

def decide(user_message: str):

    msg = user_message.lower()

    # 1. Document retrieval tool
    document_keywords = [
        "document",
        "uploaded file",
        "handbook",
        "policy",
        "according to",
        "in the pdf",
        "in the document",
        "what does the file say",
        "what does the document say"
    ]

    if any(keyword in msg for keyword in document_keywords):
        return {
            "tool": "retrieve_documents",
            "args": {
                "query": user_message,
                "k": 3
            }
        }

    # 2. Math tool
    if "add" in msg or "sum" in msg:
        return {
            "tool": "add_numbers",
            "args": {}
        }

    # 3. Web search tool
    if "search" in msg or "google" in msg:
        return {
            "tool": "search_web",
            "args": {
                "query": user_message
            }
        }

    # 4. Time tool
    if "time" in msg:
        return {
            "tool": "get_time",
            "args": {}
        }

    # 5. No tool needed
    return {
        "tool": None,
        "answer": None
    }

