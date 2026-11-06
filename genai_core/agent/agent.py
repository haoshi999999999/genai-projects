import uuid
import logging
import traceback
import inspect
import asyncio
import json

from tools.math_tools import add_numbers, multiply_numbers
from tools.search_tools import search_web
from tools.utility_tools import get_time, echo

from memory.multi_user_memory import MultiUserMemory
from memory.semantic_memory import SemanticMemory
from memory.document_ingestion import DocumentIngestion

from langchain_community.embeddings import HuggingFaceEmbeddings


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


TOOLS = {
    "add_numbers": add_numbers,
    "multiply_numbers": multiply_numbers,
    "search_web": search_web,
    "get_time": get_time,
    "echo": echo
}


TOOL_SCHEMAS = [

    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two numbers together.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer",
                        "description": "First number"
                    },
                    "b": {
                        "type": "integer",
                        "description": "Second number"
                    }
                },
                "required": ["a", "b"],
                "additionalProperties": False
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "multiply_numbers",
            "description": "Multiply two numbers together.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer"
                    },
                    "b": {
                        "type": "integer"
                    }
                },
                "required": ["a", "b"],
                "additionalProperties": False
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web for information. "
                "Use this when the user explicitly requests a search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string"
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current system date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Return the supplied text unchanged.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string"
                    }
                },
                "required": ["text"],
                "additionalProperties": False
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "retrieve_documents",
            "description": (
                "Search uploaded PDF and TXT documents for relevant "
                "information. Use this when questions concern company "
                "policies, uploaded documents, handbooks, or information "
                "that may be contained in stored documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Information to search for."
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of document chunks.",
                        "default": 3
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    }

]


class Agent:

    def __init__(self, llm):
        self.llm = llm
        self.memory = MultiUserMemory()
        self.semantic = SemanticMemory()

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.docs = DocumentIngestion(embeddings)

        self.history = {}
        self.sessions = {}

        self.personas = {
            "default": "You are a helpful AI assistant.",
            "friendly": "You respond warmly, casually, and with encouragement.",
            "technical": "You respond with precise, structured, engineering-style explanations.",
            "coach": "You respond like a motivating coach, focusing on clarity and action.",
            "concise": "You respond with minimal words, focusing only on essential points."
        }

        self.last_response = ""

    # -----------------------------
    # HELPERS
    # -----------------------------

    def _ensure_session(self, user_id, session_id):

        if user_id not in self.sessions:
            self.sessions[user_id] = {}

        if session_id not in self.sessions[user_id]:
            self.sessions[user_id][session_id] = {
                "title": "New Chat",
                "messages": [],
                "persona": "default"
            }

    def _safe_extract_content(self, response):

        try:
            # Groq/OpenAI-style dict response
            if isinstance(response, dict):
                choices = response.get("choices")

                if choices:
                    msg = choices[0].get("message")

                    if msg:
                        return msg.get("content", "")

            # Groq/OpenAI-style object response
            if hasattr(response, "choices"):
                choice0 = response.choices[0]

                if hasattr(choice0, "message"):
                    msg = choice0.message

                    if isinstance(msg, dict):
                        return msg.get("content", "")

                    if hasattr(msg, "content"):
                        return msg.content

            return ""

        except Exception:
            logger.error("Failed to parse Groq response")
            logger.error(traceback.format_exc())
            return ""

    async def _maybe_await(self, fn_or_coro, *args, **kwargs):

        result = fn_or_coro(*args, **kwargs)

        if inspect.isawaitable(result):
            return await result

        return result

    def retrieve_documents(self, user_id, query: str, k: int = 3):

        try:
            results = self.docs.search(
                query=query,
                user_id=user_id,
                k=k
            )

            if not results:
                return "No relevant documents were found."

            retrieved_content = []

            for i, doc in enumerate(results, start=1):

                source = doc.metadata.get(
                    "source", "Unknown source"
                )

                page = doc.metadata.get("page")

                page_number = (
                    page + 1 if page is not None else "Unknown"
                )

                document_type = (
                    "Shared"
                    if doc.metadata.get("is_shared")
                    else "Private"
                )

                retrieved_content.append(
                    f"[Source {i}]\n"
                    f"Document Type: {document_type}\n"
                    f"Filename: {source}\n"
                    f"Page: {page_number}\n"
                    f"Content:\n{doc.page_content}"
                )

            return "\n\n".join(retrieved_content)

        except Exception:

            logger.exception("Document retrieval failed")

            return "Document retrieval failed."

    # -----------------------------
    # SESSION MANAGEMENT
    # -----------------------------

    def create_session(self, user_id):

        session_id = str(uuid.uuid4())

        if user_id not in self.sessions:
            self.sessions[user_id] = {}

        self.sessions[user_id][session_id] = {
            "title": "New Chat",
            "messages": [],
            "persona": "default"
        }

        return session_id

    def add_to_session(self, user_id, session_id, role, content):

        self._ensure_session(user_id, session_id)

        self.sessions[user_id][session_id]["messages"].append({
            "role": role,
            "content": content
        })

    def generate_title(self, user_message):

        if len(user_message) > 40:
            return user_message[:40] + "..."

        return user_message

    # -----------------------------
    # PERSONA MANAGEMENT
    # -----------------------------

    def set_persona(self, user_id, session_id, persona_name):

        if persona_name not in self.personas:
            return False

        self._ensure_session(user_id, session_id)

        self.sessions[user_id][session_id]["persona"] = persona_name

        return True

    # -----------------------------
    # LONG-TERM FACT EXTRACTION
    # -----------------------------

    def extract_fact(self, user_id, message):

        lower = message.lower()
        marker = "remember that"

        if marker in lower:

            index = lower.find(marker)
            fact = message[index + len(marker):].strip()

            if not fact:
                return "What would you like me to remember?"

            try:
                self.memory.add_long_term(user_id, fact)
                self.semantic.add(user_id, fact)

            except Exception:
                logger.debug(
                    "Failed to persist long-term fact: %s",
                    traceback.format_exc()
                )

            return f"I will remember that: {fact}"

        return None

    # -----------------------------
    # STREAMING
    # -----------------------------

    async def stream(self, user_id, session_id, user_message):

        self._ensure_session(user_id, session_id)

        # Save user message
        self.add_to_session(
            user_id, session_id, "user", user_message
        )

        # Save user message to short-term memory
        self.memory.add_short_term(
            user_id, f"User: {user_message}"
        )

        # Update session title
        if self.sessions[user_id][session_id]["title"] == "New Chat":
            self.sessions[user_id][session_id]["title"] = (
                self.generate_title(user_message)
            )

        # -----------------------------
        # LONG-TERM MEMORY EXTRACTION
        # -----------------------------

        fact_reply = self.extract_fact(user_id, user_message)

        if fact_reply:

            self.add_to_session(
                user_id,
                session_id,
                "assistant",
                fact_reply
            )

            self.memory.add_short_term(
                user_id, f"Assistant: {fact_reply}"
            )

            self.last_response = fact_reply

            yield fact_reply
            return

        # -----------------------------
        # PREPARE LLM CONVERSATION
        # -----------------------------

        messages = self.sessions[user_id][session_id]["messages"]

        persona = self.sessions[user_id][session_id].get(
            "persona", "default"
        )

        persona_prompt = self.personas.get(
            persona, self.personas["default"]
        )

        llm_messages = [
            {"role": "system", "content": persona_prompt}
        ]

        for m in messages:

            role = m.get("role") if isinstance(m, dict) else "user"
            content = m.get("content") if isinstance(m, dict) else str(m)

            llm_messages.append({
                "role": role,
                "content": content
            })

        full_reply = ""

        # -----------------------------
        # LLM STREAMING
        # -----------------------------

        try:

            stream = await self.llm.chat.completions.create(
                model=self.llm.model_name,
                messages=llm_messages,
                stream=True
            )

            async for chunk in stream:

                try:

                    delta = None

                    if hasattr(chunk, "choices"):

                        delta = (
                            chunk.choices[0].delta
                            if hasattr(chunk.choices[0], "delta")
                            else None
                        )

                    elif isinstance(chunk, dict):

                        choices = chunk.get("choices", [])

                        if choices:
                            delta = choices[0].get("delta")

                    token = ""

                    if delta:

                        if hasattr(delta, "content"):
                            token = delta.content or ""

                        elif isinstance(delta, dict):
                            token = delta.get("content", "") or ""

                    else:

                        if hasattr(chunk, "text"):
                            token = chunk.text or ""

                        elif isinstance(chunk, dict):
                            token = chunk.get("text", "") or ""

                    if token:
                        full_reply += token
                        yield token

                except Exception:

                    logger.debug(
                        "Error parsing stream chunk: %s",
                        traceback.format_exc()
                    )

                    continue

        except Exception as exc:

            logger.error("Agent.stream failed: %s", exc)
            logger.error(traceback.format_exc())

            fallback = (
                "Sorry, the assistant is temporarily unavailable."
            )

            try:

                self.add_to_session(
                    user_id, session_id, "assistant", fallback
                )

                self.memory.add_short_term(
                    user_id, f"Assistant: {fallback}"
                )

                self.last_response = fallback

            except Exception:

                logger.debug(
                    "Failed to save fallback assistant message: %s",
                    traceback.format_exc()
                )

            yield fallback
            return

        # -----------------------------
        # SAVE FINAL ASSISTANT RESPONSE
        # -----------------------------

        try:

            self.add_to_session(
                user_id,
                session_id,
                "assistant",
                full_reply
            )

            self.memory.add_short_term(
                user_id, f"Assistant: {full_reply}"
            )

            self.last_response = full_reply

        except Exception:

            logger.error(
                "Failed to persist final assistant message: %s",
                traceback.format_exc()
            )

    async def _execute_tool_calls(self, user_id, session_id, user_message):
    
        tool_registry = {
            **TOOLS,
            "retrieve_documents": self.retrieve_documents
        }

        messages = self.sessions[user_id][session_id]["messages"]

        persona = self.sessions[user_id][session_id].get(
            "persona", "default"
        )

        persona_prompt = self.personas.get(
            persona, self.personas["default"]
        )

        context = self.memory.get_context(user_id)
        facts = self.memory.get_facts(user_id)

        system_prompt = f"""
    {persona_prompt}

    You are an intelligent AI agent with access to external tools.

    You can use tools when necessary to answer user questions.

    Instructions:

    1. Decide whether a tool is needed.
    2. Select the most appropriate tool.
    3. Do not invoke tools unnecessarily.
    4. Use document retrieval when the answer depends on uploaded documents.
    5. Use tool results as evidence when generating answers.
    6. Never invent information from retrieved documents.
    7. If document retrieval does not find relevant information,
    tell the user honestly.
    8. Answer directly when no tool is necessary.

    Conversation history:
    {context}

    Known facts:
    {facts}
    """

        llm_messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            *messages
        ]

        max_iterations = 5

        for iteration in range(max_iterations):

            response = await self.llm.chat.completions.create(
                model=self.llm.model_name,
                messages=llm_messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                parallel_tool_calls=False
            )

            assistant_message = response.choices[0].message

            tool_calls = assistant_message.tool_calls

            # No tool required: return the normal LLM response.
            if not tool_calls:

                return assistant_message.content or ""

            logger.info(
                "LLM selected %s tool(s)",
                len(tool_calls)
            )

            # Preserve the assistant's tool-call message.
            llm_messages.append(
                assistant_message.model_dump(exclude_none=True)
            )

            # Execute requested tools.
            for tool_call in tool_calls:

                tool_name = tool_call.function.name

                try:

                    args = json.loads(
                        tool_call.function.arguments
                    )

                    if not isinstance(args, dict):
                        raise ValueError("Tool arguments must be a JSON object.")

                    tool_fn = tool_registry.get(tool_name)

                    if tool_fn is None:
                        raise ValueError(
                            f"Unknown tool: {tool_name}"
                        )

                    logger.info(
                        "Executing tool: %s | Arguments: %s",
                        tool_name,
                        args
                    )

                    if tool_name == "retrieve_documents":
                        result = await self._maybe_await(
                            tool_fn,
                            user_id=user_id,
                            **args
                        )
                    else:
                        result = await self._maybe_await(
                            tool_fn,
                            **args
                        )

                    logger.info(
                        "Tool '%s' executed successfully.",
                        tool_name
                    )

                    tool_result = str(result)

                except Exception as exc:

                    logger.exception(
                        "Tool execution failed: %s",
                        tool_name
                    )

                    tool_result = (
                        f"Tool execution failed: {str(exc)}"
                    )

                # Return the execution result to the LLM.
                llm_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": tool_result
                })

        # Maximum tool iterations reached.
        logger.warning("Maximum tool iterations reached.")

        # Ask the LLM to synthesize the results without requesting more tools.
        response = await self.llm.chat.completions.create(
            model=self.llm.model_name,
            messages=llm_messages,
            tools=TOOL_SCHEMAS,
            tool_choice="none"
        )

        return response.choices[0].message.content or ""


    # -----------------------------
    # NON-STREAMING
    # -----------------------------
    async def run(self, user_id, session_id, user_message):

        self._ensure_session(user_id, session_id)

        try:

            self.add_to_session(
                user_id, session_id, "user", user_message
            )

            if self.sessions[user_id][session_id]["title"] == "New Chat":

                self.sessions[user_id][session_id]["title"] = (
                    self.generate_title(user_message)
                )

            self.memory.add_short_term(
                user_id, f"User: {user_message}"
            )

            # DOCUMENT INGESTION
            normalized_message = user_message.strip()

            if "ingest" in normalized_message.lower():

                logger.info("DOCUMENT INGESTION COMMAND DETECTED")

                # Extract everything after the first occurrence of "ingest"
                index = normalized_message.lower().find("ingest")

                file_path = normalized_message[index + len("ingest"):].strip()

                # Remove optional colon
                if file_path.startswith(":"):
                    file_path = file_path[1:].strip()

                if not file_path:
                    assistant_reply = (
                        "Please provide a file path. "
                        "Example: ingest C:\\documents\\handbook.pdf"
                    )

                else:
                    logger.info("File path: %s", file_path)

                    try:
                        result = self.docs.load(
                            file_path=file_path,
                            user_id=user_id,
                            is_shared=False
                        )

                        assistant_reply = str(result)

                        logger.info(
                            "DOCUMENT INGESTION SUCCESS: %s",
                            assistant_reply
                        )

                    except Exception as e:

                        logger.exception("Document ingestion error")

                        assistant_reply = (
                            f"Failed to ingest document: {str(e)}"
                        )

                self.add_to_session(
                    user_id,
                    session_id,
                    "assistant",
                    assistant_reply
                )

                self.memory.add_short_term(
                    user_id,
                    f"Assistant: {assistant_reply}"
                )

                self.last_response = assistant_reply

                return assistant_reply

            # FACT EXTRACTION
            fact_reply = self.extract_fact(user_id, user_message)

            if fact_reply:

                self.add_to_session(
                    user_id, session_id, "assistant", fact_reply
                )

                self.last_response = fact_reply
                return fact_reply

            # LLM-BASED TOOL ORCHESTRATION
            try:

                assistant_reply = await self._execute_tool_calls(
                    user_id,
                    session_id,
                    user_message
                )

            except Exception as exc:

                logger.exception(
                    "LLM tool orchestration failed: %s",
                    exc
                )

                assistant_reply = (
                    "Sorry, the assistant is temporarily unavailable."
                )

            # SAVE ASSISTANT RESPONSE
            self.add_to_session(
                user_id,
                session_id,
                "assistant",
                assistant_reply
            )

            self.memory.add_short_term(
                user_id,
                f"Assistant: {assistant_reply}"
            )

            self.last_response = assistant_reply

            return assistant_reply

        except Exception as exc:

            logger.exception(
                "Agent.run unexpected error: %s",
                exc
            )

            fallback = (
                "Sorry, the assistant is temporarily unavailable."
            )

            try:
                self.add_to_session(
                    user_id, session_id, "assistant", fallback
                )

                self.last_response = fallback

            except Exception:
                logger.exception("Failed to save fallback response.")

            return fallback


    