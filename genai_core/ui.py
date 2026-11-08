import streamlit as st
import requests
import json

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="GenAI Agent Chat UI")
st.title("GenAI Agent Chat UI")


# --------------------------------------------------
# AUTHENTICATION HELPER
# --------------------------------------------------

def get_auth_headers():
    return {
        "Authorization": f"Bearer {st.session_state['token']}"
    }


# --------------------------------------------------
# SSE EVENT PARSER
# --------------------------------------------------

def parse_sse_event(event_lines):
    """
    Parse one Server-Sent Event into an event name and JSON data.
    """

    event_name = "message"
    data_lines = []

    for line in event_lines:
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())

    if not data_lines:
        return event_name, {}

    data_text = "\n".join(data_lines)

    try:
        data = json.loads(data_text)
    except json.JSONDecodeError:
        data = {"message": data_text}

    return event_name, data


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

            st.session_state.clear()
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
        st.session_state["session_select"] = new_session

        st.rerun()

    except requests.exceptions.RequestException as exc:
        st.error(f"Failed to create session: {exc}")
        st.stop()


# --------------------------------------------------
# SELECT / RESTORE CHAT SESSION
# --------------------------------------------------

sessions = load_sessions()

if not sessions:
    st.session_state["session_id"] = None
    st.session_state.pop("session_select", None)

    st.sidebar.write(
        "No chats yet. Click 'New Chat' to start."
    )
    st.stop()


# The backend returns sessions ordered by most recent message activity.
# On a fresh login, select the most recently active conversation.

if (
    "session_id" not in st.session_state
    or st.session_state["session_id"] not in sessions
):
    latest_session = next(iter(sessions))
    st.session_state["session_id"] = latest_session
    st.session_state["session_select"] = latest_session


if (
    "session_select" not in st.session_state
    or st.session_state["session_select"] not in sessions
):
    st.session_state["session_select"] = st.session_state["session_id"]


def format_session_label(session_id):
    session_info = sessions[session_id]

    title = session_info.get("title") or "Untitled Chat"
    last_activity = session_info.get("last_activity")

    if last_activity:
        timestamp = str(last_activity).replace("T", " ")[:16]
        return f"{title}  ·  {timestamp}"

    return f"{title}  ·  No messages yet"


session_id = st.sidebar.selectbox(
    "Select a chat",
    options=list(sessions.keys()),
    format_func=format_session_label,
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

current_persona = sessions[session_id].get("persona", "default")

if current_persona not in persona_options:
    current_persona = "default"

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


# ==================================================
# DOCUMENT UPLOAD - SIDEBAR (MULTIPLE FILES)
# ==================================================

st.sidebar.markdown("---")
st.sidebar.subheader("Document Upload")

st.sidebar.caption(
    "Upload multiple PDF or TXT documents to your private "
    "RAG knowledge base. Only your authenticated account "
    "can retrieve these documents."
)


# Clear uploader after successful ingestion.
if st.session_state.pop("clear_document_uploader", False):
    st.session_state.pop("document_uploader", None)


# Allow multiple file selection.
uploaded_files = st.sidebar.file_uploader(
    "Choose PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    key="document_uploader"
)


if st.sidebar.button(
    "Upload and Ingest Documents",
    key="upload_document_button"
):

    if not uploaded_files:

        st.sidebar.warning(
            "Please select at least one PDF or TXT file."
        )

    else:

        total_files = len(uploaded_files)
        successful_files = []
        failed_files = []

        progress_bar = st.sidebar.progress(0)

        status_placeholder = st.sidebar.empty()

        for index, uploaded_file in enumerate(uploaded_files):

            filename = uploaded_file.name

            status_placeholder.info(
                f"Uploading {index + 1} of {total_files}: {filename}"
            )

            try:

                file_bytes = uploaded_file.getvalue()

                response = requests.post(
                    f"{API_URL}/documents/upload",
                    headers=get_auth_headers(),
                    files={
                        "file": (
                            uploaded_file.name,
                            file_bytes,
                            uploaded_file.type
                        )
                    },
                    timeout=(10, 300)
                )

                if response.status_code == 200:

                    data = response.json()

                    successful_files.append(
                        data.get("filename", filename)
                    )

                else:

                    failed_files.append(
                        {
                            "filename": filename,
                            "error": (
                                f"HTTP {response.status_code}: "
                                f"{response.text}"
                            )
                        }
                    )

            except requests.exceptions.RequestException as exc:

                failed_files.append(
                    {
                        "filename": filename,
                        "error": str(exc)
                    }
                )

            except (ValueError, KeyError) as exc:

                failed_files.append(
                    {
                        "filename": filename,
                        "error": f"Invalid backend response: {exc}"
                    }
                )

            progress_bar.progress(
                (index + 1) / total_files
            )


        # ------------------------------------------
        # UPLOAD SUMMARY
        # ------------------------------------------

        status_placeholder.empty()

        if successful_files:

            st.sidebar.success(
                f"Successfully ingested "
                f"{len(successful_files)} of {total_files} documents."
            )

            with st.sidebar.expander("Successful uploads"):

                for filename in successful_files:
                    st.write(f"✅ {filename}")


        if failed_files:

            st.sidebar.error(
                f"Failed to ingest {len(failed_files)} document(s)."
            )

            with st.sidebar.expander("Failed uploads"):

                for item in failed_files:

                    st.write(f"❌ {item['filename']}")
                    st.caption(item["error"])


        # Clear uploader only if every file succeeded.
        if not failed_files:

            st.session_state["clear_document_uploader"] = True
            st.rerun()


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
# STREAMING SEND WITH SSE
# --------------------------------------------------

if st.button(
    "Send (Streaming)",
    key="streaming_send_button"
):

    if not message.strip():
        st.warning("Please enter a message first.")

    else:

        full_text = ""
        completed = False
        had_error = False

        st.markdown("### Assistant Response")
        response_placeholder = st.empty()

        status_box = st.status(
            "Connecting to agent...",
            expanded=True
        )

        response = None

        try:
            response = requests.post(
                f"{API_URL}/stream",
                headers=get_auth_headers(),
                json={
                    "session_id": str(session_id),
                    "content": message.strip()
                },
                stream=True,
                timeout=(10, 180)
            )

            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type",
                ""
            ).lower()

            if "text/event-stream" not in content_type:
                raise ValueError(
                    f"Expected SSE response, received: {content_type}"
                )

            response.encoding = "utf-8"
            event_lines = []

            for line in response.iter_lines(decode_unicode=True):

                if line == "":

                    if not event_lines:
                        continue

                    event_name, event_data = parse_sse_event(event_lines)
                    event_lines = []

                    if event_name == "status":

                        status_message = event_data.get(
                            "message",
                            "Agent is processing..."
                        )

                        status_box.update(
                            label=status_message,
                            state="running",
                            expanded=True
                        )

                        status_box.write(status_message)

                    elif event_name == "token":

                        token = event_data.get("content", "")

                        if token:
                            full_text += token
                            response_placeholder.markdown(full_text)

                    elif event_name == "error":

                        had_error = True

                        error_message = event_data.get(
                            "message",
                            "An error occurred during processing."
                        )

                        status_box.write(f"Error: {error_message}")
                        st.warning(error_message)

                    elif event_name == "done":

                        completed = True

                        done_message = event_data.get(
                            "message",
                            "Response completed."
                        )

                        status_box.update(
                            label=done_message,
                            state="complete",
                            expanded=False
                        )

                else:

                    if line.startswith(":"):
                        continue

                    event_lines.append(line)


            # Process a final event if there was no trailing blank line.

            if event_lines:

                event_name, event_data = parse_sse_event(event_lines)

                if event_name == "token":

                    token = event_data.get("content", "")

                    if token:
                        full_text += token
                        response_placeholder.markdown(full_text)

                elif event_name == "error":

                    had_error = True

                    st.warning(
                        event_data.get(
                            "message",
                            "An error occurred."
                        )
                    )

                elif event_name == "done":

                    completed = True

                    status_box.update(
                        label=event_data.get(
                            "message",
                            "Response completed."
                        ),
                        state="complete",
                        expanded=False
                    )


            if completed:

                if full_text:

                    if had_error:
                        st.warning(
                            "The agent reported an error, "
                            "but returned a response."
                        )

                    else:
                        st.success(
                            "Response completed and saved."
                        )

                else:
                    st.warning(
                        "The agent completed without returning text."
                    )

                st.session_state["clear_message_input"] = True
                st.rerun()

            else:

                status_box.update(
                    label="Streaming ended before completion.",
                    state="error",
                    expanded=True
                )

                if full_text:

                    st.warning(
                        "A partial response was received. "
                        "The backend did not confirm completion."
                    )

                else:

                    st.error(
                        "The streaming endpoint ended without "
                        "returning a response."
                    )


        except requests.exceptions.RequestException as exc:

            status_box.update(
                label="Streaming HTTP error",
                state="error",
                expanded=True
            )

            st.error(f"Streaming HTTP error: {exc}")


        except Exception as exc:

            status_box.update(
                label="Unexpected streaming error",
                state="error",
                expanded=True
            )

            st.error(f"Unexpected streaming error: {exc}")


        finally:

            if response is not None:
                response.close()


# --------------------------------------------------
# NON-STREAMING SEND
# --------------------------------------------------

if st.button("Send"):

    if not message.strip():

        st.warning("Please enter a message first.")

    else:

        try:
            response = requests.post(
                f"{API_URL}/message/send",
                headers=get_auth_headers(),
                json={
                    "session_id": str(session_id),
                    "content": message.strip()
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

                    st.error(
                        f"Invalid response from backend: {exc}"
                    )

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


# Group messages into conversation exchanges.
# Reverse the exchanges, but preserve user-before-assistant order.

conversation_turns = []
current_turn = []

for msg in msgs:

    if msg["role"] == "user":

        if current_turn:
            conversation_turns.append(current_turn)

        current_turn = [msg]

    else:

        current_turn.append(msg)

if current_turn:
    conversation_turns.append(current_turn)


# Display newest conversation exchanges first.

for turn in reversed(conversation_turns):

    for msg in turn:

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