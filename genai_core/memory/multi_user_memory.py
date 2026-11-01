import json
from pathlib import Path


class MultiUserMemory:
    def __init__(self, path=None):
        # Resolve the default path relative to this Python file.
        if path is None:
            self.path = Path(__file__).resolve().parent / "memory.json"
        else:
            self.path = Path(path).expanduser().resolve()

        # Ensure the parent directory exists.
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.data = {}

        # Load existing memory or create a new JSON file.
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def ensure_user(self, user_id):
        if user_id not in self.data:
            self.data[user_id] = {
                "short_term": [],
                "long_term": []
            }

    def add_short_term(self, user_id, msg):
        self.ensure_user(user_id)

        short_term = self.data[user_id]["short_term"]
        short_term.append(msg)

        # Keep only the 10 most recent short-term entries.
        if len(short_term) > 10:
            short_term.pop(0)

        self.save()

    def add_long_term(self, user_id, fact):
        self.ensure_user(user_id)
        self.data[user_id]["long_term"].append(fact)
        self.save()

    def get_context(self, user_id):
        self.ensure_user(user_id)
        return "\n".join(self.data[user_id]["short_term"])

    def get_facts(self, user_id):
        self.ensure_user(user_id)
        return "\n".join(self.data[user_id]["long_term"])

