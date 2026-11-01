import json
import os

class Memory:
    def __init__(self, path="genai_agent/memory/memory.json"):
        self.path = path
        self.short_term = []
        self.long_term = []

        # Load persistent memory
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                data = json.load(f)
                self.long_term = data.get("long_term", [])
        else:
            # Create file if missing
            with open(self.path, "w") as f:
                json.dump({"long_term": []}, f)

    def save(self):
        with open(self.path, "w") as f:
            json.dump({"long_term": self.long_term}, f, indent=2)

    def add_short_term(self, message):
        self.short_term.append(message)
        if len(self.short_term) > 10:
            self.short_term.pop(0)

    def add_long_term(self, fact):
        self.long_term.append(fact)
        self.save()  # persist immediately

    def get_context(self):
        return "\n".join(self.short_term)

    def get_facts(self):
        return "\n".join(self.long_term)
