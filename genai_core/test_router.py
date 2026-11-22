from agent.router import decide

tests = [
    "What does the uploaded document say about vacation policy?",
    "Search the handbook for security requirements.",
    "Add 10 and 20.",
    "Google the latest AI news.",
    "What time is it?",
    "Explain how RAG works."
]

for question in tests:
    print(f"\nQuestion: {question}")
    print(f"Decision: {decide(question)}")