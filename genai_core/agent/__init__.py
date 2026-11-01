import uuid

class Agent:
    def __init__(self, llm):
        self.llm = llm
        self.sessions = {}  # { user_id: { session_id: {title, messages, persona} } }
        ...
    
    # -----------------------------
    # Session Management
    # -----------------------------
    def create_session(self, user_id):
        session_id = str(uuid.uuid4())
        if user_id not in self.sessions:
            self.sessions[user_id] = {}

        self.sessions[user_id][session_id] = {
            "title": "New Chat",
            "messages": [],
            "persona": "default"   # required for Option D
        }
        return session_id

    def add_to_session(self, user_id, session_id, role, content):
        self.sessions[user_id][session_id]["messages"].append({
            "role": role,
            "content": content
        })

    def generate_title(self, user_message):
        if len(user_message) > 40:
            return user_message[:40] + "..."
        return user_message
