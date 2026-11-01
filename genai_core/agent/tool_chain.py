# genai_core/agent/tool_chain.py

# 1. Define your tools here (no circular imports)
TOOLS = {
    "add_numbers": lambda x, y: x + y,
    "multiply_numbers": lambda x, y: x * y,
    # Add more tools here if needed
}

def run_tool_chain(llm, decision, memory, user_id):
    steps = decision["tool"]
    results = []

    for step in steps:
        tool_name = step["tool"]
        args = step["args"]

        tool_fn = TOOLS.get(tool_name)
        if tool_fn is None:
            raise ValueError(f"Unknown tool: {tool_name}")

        result = tool_fn(**args)
        results.append({"tool": tool_name, "result": result})

    prompt = f"""
Tool chain executed:
{results}

Now synthesize a final answer for the user.
"""

    reply = llm.invoke(prompt)
    content = reply.content
    memory.add_short_term(user_id, f"Assistant: {content}")
    return content
