import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="GenAI Agent Chat UI")
st.title("GenAI Agent Chat UI")

# -----------------------------
# LOGIN FORM
# -----------------------------
if "email" not in st.session_state:
    st.subheader("Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        try:
            resp = requests.post(f"{API_URL}/login", json={"email": email, "password": password}, timeout=10)
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not connect to the backend: {exc}")
            st.stop()

        if resp.status_code == 200:
            # Clear all stale widget state to avoid "list" or other stale values
            st.session_state.clear()
            st.session_state["email"] = email
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid email or password")

    st.stop()


# -----------------------------
# Load sessions for this user
# -----------------------------
def load_sessions():
    try:
        resp = requests.get(f"{API_URL}/session/list?email={st.session_state['email']}", timeout=10)
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not load sessions: {exc}")
        return {}

    if resp.status_code == 200:
        return resp.json()
    st.write("Session list error:", resp.status_code, resp.text)
    return {}


st.sidebar.title("Your Chats")

# -----------------------------
# Create new chat session
# -----------------------------
if st.sidebar.button("New Chat"):
    try:
        new_session = requests.post(
            f"{API_URL}/session/create",
            json={"email": st.session_state["email"], "persona": "default", "title": "New Chat"},
            timeout=10
        ).json()["session_id"]
    except Exception as exc:
        st.error(f"Failed to create session: {exc}")
        st.stop()

    st.session_state["session_id"] = new_session
    st.rerun()
    st.stop()

# -----------------------------
# Select chat session
# -----------------------------
sessions = load_sessions()

# Reset stale widget AND stale session_id
st.session_state.pop("session_select", None)
if st.session_state.get("session_id") == "list":
    st.session_state["session_id"] = None

# If no sessions exist, stop UI
if not sessions:
    st.session_state["session_id"] = None
    st.sidebar.write("No chats yet. Click 'New Chat' to start.")
    st.stop()

# Ensure session_id exists and is valid
if "session_id" not in st.session_state or st.session_state["session_id"] not in sessions:
    st.session_state["session_id"] = list(sessions.keys())[0]

session_id = st.sidebar.selectbox(
    "Select a chat",
    options=list(sessions.keys()),
    format_func=lambda sid: sessions[sid]["title"],
    index=list(sessions.keys()).index(st.session_state["session_id"]),
    key="session_select"
)

st.session_state["session_id"] = session_id

# -----------------------------
# Persona selector
# -----------------------------
st.sidebar.subheader("Persona")

persona_options = ["default", "friendly", "technical", "coach", "concise"]

current_persona = sessions[session_id]["persona"]

new_persona = st.sidebar.selectbox(
    "Choose persona",
    persona_options,
    index=persona_options.index(current_persona)
)

if new_persona != current_persona:
    try:
        requests.post(f"{API_URL}/session/persona", json={"session_id": session_id, "persona": new_persona}, timeout=10)
    except requests.exceptions.RequestException as exc:
        st.error(f"Failed to update persona: {exc}")
    st.rerun()

# -----------------------------
# MESSAGE INPUT
# -----------------------------
#message = st.text_input("Your message")

# Clear the input on the next rerun after successful submission
if st.session_state.pop("clear_message_input", False):
    st.session_state["message_input"] = ""

message = st.text_input(
    "Your message",
    key="message_input"
)

# -----------------------------
# STREAMING SEND
# -----------------------------
if st.button("Send (Streaming)"):
    if message.strip():
        placeholder = st.empty()
        full_text = ""

        try:
            response = requests.post(
                f"{API_URL}/stream",
                json={"session_id": session_id, "content": message},
                stream=True,
                timeout=120
            )

            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    token = chunk.decode("utf-8")
                    full_text += token
                    placeholder.markdown(full_text)

            # Clear input only after successful streaming
            st.session_state["clear_message_input"] = True
            st.rerun()

        except requests.exceptions.RequestException as exc:
            st.error(f"Streaming failed: {exc}")

# -----------------------------
# NON-STREAMING SEND
# -----------------------------
if st.button("Send"):
    if message.strip():
        try:
            response = requests.post(
                f"{API_URL}/message/send",
                json={"session_id": str(session_id), "content": message},
                timeout=120
            )

        except requests.exceptions.RequestException as exc:
            st.error(f"Could not connect to the backend: {exc}")

        else:
            if not response.ok:
                st.error(
                    f"Backend error ({response.status_code}): {response.text}"
                )
            else:
                try:
                    data = response.json()

                except ValueError as exc:
                    st.error(f"Invalid response from backend: {exc}")

                else:
                    if "reply" not in data:
                        st.error(f"Unexpected backend response: {data}")
                    else:
                        # Clear input after successful backend response
                        st.session_state["clear_message_input"] = True
                        st.rerun()

# -----------------------------
# Conversation History
# -----------------------------
st.markdown("### Conversation History")

try:
    msgs_resp = requests.get(f"{API_URL}/messages/{session_id}", timeout=10)
    if msgs_resp.ok:
        msgs = msgs_resp.json()
    else:
        st.error(f"Failed to load messages: {msgs_resp.status_code} {msgs_resp.text}")
        msgs = []
except requests.exceptions.RequestException as exc:
    st.error(f"Could not load messages: {exc}")
    msgs = []

for msg in msgs:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Assistant:** {msg['content']}")

# -----------------------------
# Clear Conversation
# -----------------------------
if st.button("Clear Conversation"):
    clear_payload = {
        "session_id": session_id,
        "content": ""
    }

    try:
        r = requests.post(f"{API_URL}/session/clear", json=clear_payload, timeout=10)
    except requests.exceptions.RequestException as exc:
        st.error(f"Failed to clear conversation: {exc}")
    else:
        if r.status_code == 200:
            st.success("Conversation cleared.")
            st.rerun()
        else:
            st.error(f"Failed to clear conversation: {r.status_code} {r.text}")





