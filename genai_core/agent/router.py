import json
import inspect

SYSTEM_PROMPT = """
You are an AI agent with tools.

Available tools:
- add_numbers(a, b)
- search_web(query)
- get_time()

When a tool is needed, respond ONLY in JSON:
{"tool": "<tool_name>", "args": {...}}

If no tool is needed:
{"tool": null, "answer": "<final_answer>"}
"""

def decide(user_message: str):
    msg = user_message.lower()

    if "add" in msg or "sum" in msg:
        return {"tool": "add_numbers", "args": {}}

    if "search" in msg or "google" in msg:
        return {"tool": "search_web", "args": {"query": user_message}}

    if "time" in msg:
        return {"tool": "get_time", "args": {}}

    return {"tool": None, "answer": None}
