import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="GenAI Agent Chat UI")
st.title("GenAI Agent Chat UI")


# --------------------------------------------------
# Authentication helper
# --------------------------------------------------

def get_auth_headers():
    return {
        "Authorization": f"Bearer {st.session_state['token']}"
    }


# --------------------------------------------------
# LOGIN FORM
# --------------------------------------------------

if "token" not in st.session_state:

    st.subheader("Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        try:
            resp = requests.post(
                f"{API_URL}/login",
                json={
                    "email": email,
                    "password": password
                },
                timeout=10
            )

        except requests.exceptions.RequestException as exc:
            st.error(f"Could not connect to the backend: {exc}")
            st.stop()

        if resp.status_code == 200:

            data = resp.json()

            # Clear previous user's application state
            st.session_state.clear()

            # Save authenticated user information
            st.session_state["email"] = email
            st.session_state["token"] = data["access_token"]

            st.success("Login successful")
            st.rerun()

        else:
            st.error(f"Login failed: {resp.text}")

    st.stop()


# --------------------------------------------------
# LOGOUT
# --------------------------------------------------

if st.sidebar.button("Logout"):

    st.session_state.clear()
    st.rerun()


# --------------------------------------------------
# LOAD SESSIONS FOR CURRENT USER
# --------------------------------------------------

def load_sessions():

    try:
        resp = requests.get(
            f"{API_URL}/session/list",
            headers=get_auth_headers(),
            timeout=10
        )

    except requests.exceptions.RequestException as exc:
        st.error(f"Could not load sessions: {exc}")
        return {}

    if resp.status_code == 200:
        return resp.json()

    st.error(
        f"Session list error: {resp.status_code} {resp.text}"
    )

    return {}


st.sidebar.title("Your Chats")
st.sidebar.caption(f"Logged in as: {st.session_state['email']}")


# --------------------------------------------------
# CREATE NEW CHAT SESSION
# --------------------------------------------------

if st.sidebar.button("New Chat"):

    try:
        response = requests.post(
            f"{API_URL}/session/create",
            headers=get_auth_headers(),
            json={
                "persona": "default",
                "title": "New Chat"
            },
            timeout=10
        )

        response.raise_for_status()

        new_session = str(response.json()["session_id"])

        st.session_state["session_id"] = new_session

        st.rerun()

    except requests.exceptions.RequestException as exc:
        st.error(f"Failed to create session: {exc}")
        st.stop()


# --------------------------------------------------
# SELECT CHAT SESSION
# --------------------------------------------------

sessions = load_sessions()

if not sessions:

    st.session_state["session_id"] = None

    st.sidebar.write(
        "No chats yet. Click 'New Chat' to start."
    )

    st.stop()


# Ensure selected session belongs to current user

if (
    "session_id" not in st.session_state
    or st.session_state["session_id"] not in sessions
):
    st.session_state["session_id"] = list(sessions.keys())[0]


session_id = st.sidebar.selectbox(
    "Select a chat",
    options=list(sessions.keys()),
    format_func=lambda sid: sessions[sid]["title"],
    index=list(sessions.keys()).index(
        st.session_state["session_id"]
    ),
    key="session_select"
)

st.session_state["session_id"] = session_id


# --------------------------------------------------
# PERSONA SELECTOR
# --------------------------------------------------

st.sidebar.subheader("Persona")

persona_options = [
    "default",
    "friendly",
    "technical",
    "coach",
    "concise"
]

current_persona = sessions[session_id]["persona"]

new_persona = st.sidebar.selectbox(
    "Choose persona",
    persona_options,
    index=persona_options.index(current_persona)
)

if new_persona != current_persona:

    try:
        response = requests.post(
            f"{API_URL}/session/persona",
            headers=get_auth_headers(),
            json={
                "session_id": session_id,
                "persona": new_persona
            },
            timeout=10
        )

        response.raise_for_status()

        st.rerun()

    except requests.exceptions.RequestException as exc:
        st.error(f"Failed to update persona: {exc}")


# --------------------------------------------------
# MESSAGE INPUT
# --------------------------------------------------

if st.session_state.pop("clear_message_input", False):
    st.session_state["message_input"] = ""

message = st.text_input(
    "Your message",
    key="message_input"
)


# --------------------------------------------------
# STREAMING SEND
# --------------------------------------------------

if st.button("Send (Streaming)"):

    if message.strip():

        placeholder = st.empty()
        full_text = ""

        try:
            response = requests.post(
                f"{API_URL}/stream",
                headers=get_auth_headers(),
                json={
                    "session_id": session_id,
                    "content": message
                },
                stream=True,
                timeout=120
            )

            response.raise_for_status()

            for chunk in response.iter_content(
                chunk_size=None
            ):

                if chunk:
                    token = chunk.decode("utf-8")
                    full_text += token
                    placeholder.markdown(full_text)

            st.session_state["clear_message_input"] = True
            st.rerun()

        except requests.exceptions.RequestException as exc:
            st.error(f"Streaming failed: {exc}")


# --------------------------------------------------
# NON-STREAMING SEND
# --------------------------------------------------

if st.button("Send"):

    if message.strip():

        try:
            response = requests.post(
                f"{API_URL}/message/send",
                headers=get_auth_headers(),
                json={
                    "session_id": str(session_id),
                    "content": message
                },
                timeout=120
            )

        except requests.exceptions.RequestException as exc:
            st.error(f"Could not connect to the backend: {exc}")

        else:

            if not response.ok:

                st.error(
                    f"Backend error ({response.status_code}): "
                    f"{response.text}"
                )

            else:

                try:
                    data = response.json()

                except ValueError as exc:
                    st.error(f"Invalid response from backend: {exc}")

                else:

                    if "reply" not in data:
                        st.error(
                            f"Unexpected backend response: {data}"
                        )

                    else:
                        st.session_state["clear_message_input"] = True
                        st.rerun()


# --------------------------------------------------
# CONVERSATION HISTORY
# --------------------------------------------------

st.markdown("### Conversation History")

try:
    msgs_resp = requests.get(
        f"{API_URL}/messages/{session_id}",
        headers=get_auth_headers(),
        timeout=10
    )

    if msgs_resp.ok:
        msgs = msgs_resp.json()

    else:
        st.error(
            f"Failed to load messages: "
            f"{msgs_resp.status_code} {msgs_resp.text}"
        )

        msgs = []

except requests.exceptions.RequestException as exc:

    st.error(f"Could not load messages: {exc}")
    msgs = []


for msg in msgs:

    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")

    else:
        st.markdown(f"**Assistant:** {msg['content']}")


# --------------------------------------------------
# CLEAR CONVERSATION
# --------------------------------------------------

if st.button("Clear Conversation"):

    try:
        response = requests.post(
            f"{API_URL}/session/clear",
            headers=get_auth_headers(),
            json={
                "session_id": session_id
            },
            timeout=10
        )

    except requests.exceptions.RequestException as exc:

        st.error(f"Failed to clear conversation: {exc}")

    else:

        if response.status_code == 200:

            st.success("Conversation cleared.")
            st.rerun()

        else:

            st.error(
                f"Failed to clear conversation: "
                f"{response.status_code} {response.text}"
            )