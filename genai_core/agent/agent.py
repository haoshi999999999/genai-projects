import uuid
import logging
import traceback
import inspect
import asyncio

from tools.math_tools import add_numbers
from tools.search_tools import search_web
from tools.utility_tools import get_time

from memory.multi_user_memory import MultiUserMemory
from agent.tool_chain import run_tool_chain
from memory.semantic_memory import SemanticMemory
from memory.document_ingestion import DocumentIngestion

from langchain_community.embeddings import HuggingFaceEmbeddings
from agent.router import decide


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


TOOLS = {
    "add_numbers": add_numbers,
    "search_web": search_web,
    "get_time": get_time
}


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

        if "remember that" in lower:

            fact = message.split("remember that", 1)[1].strip()

            try:
                self.memory.add_long_term(user_id, fact)
                self.semantic.add(fact)

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

            messages = self.sessions[user_id][session_id]["messages"]

            self.memory.add_short_term(
                user_id, f"User: {user_message}"
            )

            # -----------------------------
            # DOCUMENT INGESTION
            # -----------------------------

            if user_message.lower().startswith("ingest "):

                file_path = user_message.split("ingest ", 1)[1].strip()

                try:

                    result = self.docs.load(file_path)

                except Exception as e:

                    logger.exception("Document ingestion error")

                    return f"Failed to ingest document: {str(e)}"

                self.add_to_session(
                    user_id, session_id, "assistant", result
                )

                self.last_response = result

                return result

            # -----------------------------
            # FACT EXTRACTION
            # -----------------------------

            fact_reply = self.extract_fact(user_id, user_message)

            if fact_reply:

                self.add_to_session(
                    user_id, session_id, "assistant", fact_reply
                )

                self.last_response = fact_reply

                return fact_reply

            # -----------------------------
            # TOOL ROUTING
            # -----------------------------

            try:

                decision = decide(user_message)

                if inspect.isawaitable(decision):
                    decision = await decision

            except Exception as e:

                logger.error("Router decide() failed: %s", e)
                logger.error(traceback.format_exc())

                decision = {
                    "tool": None,
                    "args": {}
                }

            # =================================================
            # NO-TOOL PATH: SEMANTIC MEMORY + DOCUMENT RAG
            # =================================================

            if decision.get("tool") is None:

                context = self.memory.get_context(user_id)
                facts = self.memory.get_facts(user_id)

                # -----------------------------
                # RETRIEVE SEMANTIC MEMORY
                # -----------------------------

                try:

                    semantic_hits = self.semantic.search(user_message)
                    semantic_context = "\n".join(semantic_hits)

                except Exception:

                    semantic_context = ""

                    logger.exception(
                        "Semantic memory search failed"
                    )

                # -----------------------------
                # RETRIEVE DOCUMENT CONTENT
                # -----------------------------

                try:

                    doc_hits = self.docs.search(
                        user_message, k=3
                    )

                    # Temporary RAG debugging
                    logger.info("RAG retrieved %s chunks", len(doc_hits))

                    for i, hit in enumerate(doc_hits, start=1):
                        logger.info(
                            "RAG result %s | metadata=%s | content=%s",
                            i,
                            hit.metadata,
                            hit.page_content[:300]
                        )

                    doc_context_parts = []

                    for i, hit in enumerate(doc_hits, start=1):

                        source = hit.metadata.get(
                            "source", "Unknown source"
                        )

                        # PyPDFLoader uses zero-based page numbers.
                        page = hit.metadata.get("page")

                        page_number = (
                            page + 1
                            if page is not None
                            else "Unknown"
                        )

                        doc_context_parts.append(
                            f"[Source {i}]\n"
                            f"Filename: {source}\n"
                            f"Page: {page_number}\n"
                            f"Content:\n{hit.page_content}"
                        )

                    doc_context = "\n\n".join(doc_context_parts)

                except Exception:

                    doc_context = ""

                    logger.exception(
                        "Document search failed"
                    )

                # -----------------------------
                # BUILD SYSTEM PROMPT
                # -----------------------------

                persona = self.sessions[user_id][session_id].get(
                    "persona", "default"
                )

                persona_prompt = self.personas.get(
                    persona, self.personas["default"]
                )

                system_prompt = f"""
{persona_prompt}

You are an AI assistant with access to retrieved document content.

Instructions for document-related questions:

1. Use the relevant document context as the source of truth.
2. Do not invent facts or policy details that are not supported by the retrieved content.
3. If the retrieved content does not contain the answer, clearly say that
   the available documents do not provide that information.
4. When answering using document content, cite the source using its
   source number, filename, and page number.
5. Do not fabricate source references.

Conversation history:
{context}

Relevant semantic memory:
{semantic_context}

Relevant document context:
{doc_context}

Known facts:
{facts}

Respond naturally and clearly.
"""

                # -----------------------------
                # CALL LLM
                # -----------------------------

                try:

                    response = await self.llm.chat.completions.create(
                        model=self.llm.model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            *messages
                        ]
                    )

                    assistant_reply = (
                        self._safe_extract_content(response) or ""
                    )

                except Exception as exc:

                    logger.error(
                        "LLM call failed (no-tool path): %s",
                        exc
                    )

                    logger.error(traceback.format_exc())

                    assistant_reply = (
                        "Sorry, the assistant is temporarily unavailable."
                    )

                # -----------------------------
                # SAVE ASSISTANT RESPONSE
                # -----------------------------

                self.add_to_session(
                    user_id,
                    session_id,
                    "assistant",
                    assistant_reply
                )

                self.memory.add_short_term(
                    user_id, f"Assistant: {assistant_reply}"
                )

                self.last_response = assistant_reply

                return assistant_reply

            # =================================================
            # TOOL PATH
            # =================================================

            # -----------------------------
            # TOOL CHAIN
            # -----------------------------

            if isinstance(decision.get("tool"), list):

                try:

                    result = await run_tool_chain(
                        self.llm,
                        decision,
                        self.memory,
                        user_id
                    )

                except Exception as exc:

                    logger.error(
                        "run_tool_chain failed: %s",
                        exc
                    )

                    logger.error(traceback.format_exc())

                    result = "Tool chain failed."

                self.last_response = result

                return result

            # -----------------------------
            # SINGLE TOOL EXECUTION
            # -----------------------------

            tool_name = decision.get("tool")
            args = decision.get("args", {})

            tool_fn = TOOLS.get(tool_name)

            if tool_fn is None:

                result = f"Unknown tool: {tool_name}"

            else:

                try:

                    result = await self._maybe_await(
                        tool_fn, **args
                    )

                except Exception as exc:

                    logger.error(
                        "Tool execution failed: %s",
                        exc
                    )

                    logger.error(traceback.format_exc())

                    result = "Tool execution failed."

            # -----------------------------
            # TOOL RESULT SYNTHESIS
            # -----------------------------

            context = self.memory.get_context(user_id)
            facts = self.memory.get_facts(user_id)

            synthesis_prompt = f"""
You are a helpful AI assistant.

Conversation history:
{context}

Known facts:
{facts}

The user asked: {user_message}

You executed tool '{tool_name}' with args: {args}
Tool result: {result}

Now respond naturally to the user using the tool result.
"""

            try:

                response = await self.llm.chat.completions.create(
                    model=self.llm.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": synthesis_prompt
                        },
                        *messages
                    ]
                )

                assistant_reply = (
                    self._safe_extract_content(response) or ""
                )

            except Exception as exc:

                logger.error(
                    "LLM call failed (tool synthesis): %s",
                    exc
                )

                logger.error(traceback.format_exc())

                assistant_reply = (
                    "Sorry, the assistant is temporarily unavailable."
                )

            # -----------------------------
            # SAVE ASSISTANT RESPONSE
            # -----------------------------

            self.add_to_session(
                user_id,
                session_id,
                "assistant",
                assistant_reply
            )

            self.memory.add_short_term(
                user_id, f"Assistant: {assistant_reply}"
            )

            self.last_response = assistant_reply

            return assistant_reply

        # -----------------------------
        # OUTER ERROR HANDLING
        # -----------------------------

        except Exception as exc:

            logger.error(
                "Agent.run unexpected error: %s",
                exc
            )

            logger.error(traceback.format_exc())

            fallback = (
                "Sorry, the assistant is temporarily unavailable."
            )

            try:

                self.add_to_session(
                    user_id,
                    session_id,
                    "assistant",
                    fallback
                )

                self.last_response = fallback

            except Exception:

                logger.debug(
                    "Failed to save fallback after unexpected error."
                )

            return fallback