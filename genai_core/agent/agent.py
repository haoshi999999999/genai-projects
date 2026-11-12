import uuid
import logging
import traceback
import inspect
import json
import re
import asyncio

from pathlib import Path

from tools.math_tools import add_numbers, multiply_numbers
from tools.search_tools import search_web
from tools.utility_tools import get_time, echo

from memory.multi_user_memory import MultiUserMemory
from memory.semantic_memory import SemanticMemory
from memory.document_ingestion import DocumentIngestion

from langchain_community.embeddings import HuggingFaceEmbeddings


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# --------------------------------------------------
# TOOL REGISTRY
# --------------------------------------------------

TOOLS = {
    "add_numbers": add_numbers,
    "multiply_numbers": multiply_numbers,
    "search_web": search_web,
    "get_time": get_time,
    "echo": echo
}


# --------------------------------------------------
# TOOL SCHEMAS
# --------------------------------------------------

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
                "Use this when the user explicitly requests a search "
                "or needs current external information."
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
                        "description": "Number of document chunks to retrieve.",
                        "default": 5
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    }

]


# ==================================================
# AGENT CLASS
# ==================================================

class Agent:

    def __init__(self, llm):

        self.llm = llm

        # Conversation and long-term memory.
        self.memory = MultiUserMemory()
        self.semantic = SemanticMemory()

        # Document embeddings.
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Document ingestion and FAISS retrieval.
        self.docs = DocumentIngestion(embeddings)

        # User-specific conversation sessions.
        self.history = {}
        self.sessions = {}

        # Response personas.
        self.personas = {
            "default": "You are a helpful AI assistant.",
            "friendly": "You respond warmly, casually, and with encouragement.",
            "technical": "You respond with precise, structured, engineering-style explanations.",
            "coach": "You respond like a motivating coach, focusing on clarity and action.",
            "concise": "You respond with minimal words, focusing only on essential points."
        }

        # Maximum number of consecutive tool-selection rounds.
        self.max_tool_iterations = 3

    # ==================================================
    # HELPERS
    # ==================================================

    def _ensure_session(self, user_id, session_id):

        if user_id not in self.sessions:
            self.sessions[user_id] = {}

        if session_id not in self.sessions[user_id]:

            self.sessions[user_id][session_id] = {
                "title": "New Chat",
                "messages": [],
                "persona": "default"
            }

    async def _maybe_await(self, fn_or_coro, *args, **kwargs):

        result = fn_or_coro(*args, **kwargs)

        if inspect.isawaitable(result):
            return await result

        return result

    def _emit_event(self, event, data):

        return {
            "event": event,
            "data": data
        }

    def _get_tool_status(self, tool_name):

        statuses = {
            "search_web": "Searching the web...",
            "retrieve_documents": "Searching uploaded documents...",
            "add_numbers": "Calculating...",
            "multiply_numbers": "Calculating...",
            "get_time": "Checking the current time...",
            "echo": "Processing your request..."
        }

        return statuses.get(
            tool_name,
            f"Executing {tool_name}..."
        )

    def _get_tool_display_name(self, tool_name):

        names = {
            "search_web": "Web search",
            "retrieve_documents": "Document retrieval",
            "add_numbers": "Addition",
            "multiply_numbers": "Multiplication",
            "get_time": "Time lookup",
            "echo": "Utility tool"
        }

        return names.get(tool_name, tool_name)

    # ==================================================
    # DOCUMENT INGESTION HELPERS
    # ==================================================

    def _parse_ingestion_command(self, message):
        """
        Recognize explicit document ingestion commands.
        """

        message = message.strip()

        if re.fullmatch(
            r"ingest\s*:?",
            message,
            flags=re.IGNORECASE
        ):
            return ""

        match = re.match(
            r"^\s*ingest\s*:?\s+(.+?)\s*$",
            message,
            flags=re.IGNORECASE
        )

        if not match:
            return None

        file_path = match.group(1).strip()

        if (
            len(file_path) >= 2
            and file_path[0] == file_path[-1]
            and file_path[0] in ('"', "'")
        ):
            file_path = file_path[1:-1].strip()

        if Path(file_path).suffix.lower() not in (".pdf", ".txt"):
            return None

        return file_path

    def _ingest_document(
        self,
        user_id,
        file_path,
        source_name=None,
        document_name=None,
        document_version=None,
        effective_date=None,
        authority=None,
        owner_department=None
    ):
        """
        Ingest a private PDF or TXT document
        for the authenticated user.
        """

        if not file_path:

            return (
                "Please provide the full path to a PDF or TXT file.\n\n"
                "Example:\n"
                'ingest "C:\\documents\\handbook.pdf"'
            )

        try:

            path = Path(file_path).expanduser()

            if not path.exists():

                return (
                    f"File not found: {path}\n\n"
                    "Please check the path and try again."
                )

            if not path.is_file():

                return (
                    "The specified path is not a file. "
                    "Please provide a PDF or TXT file."
                )

            if path.suffix.lower() not in (".pdf", ".txt"):

                return (
                    "Unsupported file type. "
                    "Only PDF and TXT documents are supported."
                )

            if source_name:
                source_name = Path(source_name).name
            else:
                source_name = path.name

            result = self.docs.load(
                file_path=str(path),
                user_id=user_id,
                is_shared=False,
                source_name=source_name,
                document_name=document_name,
                document_version=document_version,
                effective_date=effective_date,
                authority=authority,
                owner_department=owner_department
            )

            # ------------------------------------------
            # VALIDATE INGESTION RESULT
            # ------------------------------------------

            if not isinstance(result, dict):

                logger.error(
                    "Unexpected document ingestion result type | "
                    "User: %s | File: %s | Result: %s",
                    user_id,
                    source_name,
                    result
                )

                return (
                    "Document ingestion returned an unexpected result. "
                    "Please check the application logs."
                )

            status = result.get("status")

            logger.info(
                "Document ingestion result | "
                "User: %s | File: %s | Status: %s",
                user_id,
                source_name,
                status
            )

            # ------------------------------------------
            # HANDLE SUCCESSFUL INGESTION
            # ------------------------------------------

            if status == "ingested":

                return (
                    f"Document uploaded and indexed successfully: "
                    f"{source_name}"
                )

            elif status == "skipped":

                return (
                    "Document already exists with identical content "
                    "and metadata. No changes were needed: "
                    f"{source_name}"
                )

            elif status == "metadata_updated":

                return (
                    "Document metadata updated successfully. "
                    "No duplicate chunks were created: "
                    f"{source_name}"
                )

            # ------------------------------------------
            # HANDLE UNEXPECTED STATUS
            # ------------------------------------------

            else:

                logger.error(
                    "Unexpected document ingestion status | "
                    "User: %s | File: %s | Result: %s",
                    user_id,
                    source_name,
                    result
                )

                return (
                    "Document ingestion returned an unexpected status: "
                    f"{source_name}"
                )

        except Exception:

            logger.exception(
                "Document ingestion failed for user %s",
                user_id
            )

            return (
                "Document ingestion failed. "
                "Please check the document and try again."
            )

    # ==================================================
    # DOCUMENT RETRIEVAL
    # ==================================================

    def retrieve_documents(self, user_id, query: str, k: int = 5):

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
                    "source",
                    "Unknown source"
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

    # ==================================================
    # DOCUMENT QUERY DETECTION
    # ==================================================

    def _should_retrieve_documents(self, user_message):
        """
        Identify questions that are likely to require
        information from uploaded company documents.
        """

        text = user_message.lower()

        explicit_terms = [
            "uploaded document",
            "uploaded file",
            "uploaded documents",
            "employee handbook",
            "company handbook",
            "company policy",
            "company policies",
            "vacation policy",
            "leave policy",
            "hr policy",
            "according to the document",
            "according to the handbook",
            "according to the policy"
        ]

        if any(term in text for term in explicit_terms):
            return True

        policy_topics = [
            "vacation",
            "pto",
            "paid time off",
            "annual leave",
            "sick leave",
            "holiday allowance",
            "employee benefits",
            "carry over",
            "carryover",
            "unused leave",
            "unused vacation"
        ]

        policy_actions = [
            "days",
            "allowance",
            "request",
            "requests",
            "approval",
            "approve",
            "approved",
            "submit",
            "submission",
            "balance",
            "policy",
            "annually",
            "annual",
            "employees",
            "employee"
        ]

        has_policy_topic = any(
            term in text for term in policy_topics
        )

        has_policy_context = any(
            term in text for term in policy_actions
        )

        return has_policy_topic and has_policy_context

    # ==================================================
    # SESSION MANAGEMENT
    # ==================================================

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

    # ==================================================
    # PERSONA MANAGEMENT
    # ==================================================

    def set_persona(self, user_id, session_id, persona_name):

        if persona_name not in self.personas:
            return False

        self._ensure_session(user_id, session_id)

        self.sessions[user_id][session_id]["persona"] = persona_name

        return True

    # ==================================================
    # LONG-TERM FACT EXTRACTION
    # ==================================================

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

    # ==================================================
    # SYSTEM PROMPT
    # ==================================================

    def _build_system_prompt(self, user_id, session_id):

        persona = self.sessions[user_id][session_id].get(
            "persona",
            "default"
        )

        persona_prompt = self.personas.get(
            persona,
            self.personas["default"]
        )

        context = self.memory.get_context(user_id)
        facts = self.memory.get_facts(user_id)

        return f"""
{persona_prompt}

You are an intelligent AI agent with access to external tools.

You can use tools when necessary to answer user questions.

Instructions:

1. Decide whether a tool is needed.
2. Select the most appropriate tool.
3. Do not invoke tools unnecessarily.
4. Use document retrieval when the answer depends on uploaded documents.
5. Use web search when the user explicitly requests current web information.
6. Use tool results as evidence when generating answers.
7. Never invent information from retrieved documents.
8. If document retrieval does not find relevant information,
   tell the user honestly.
9. Answer directly when no tool is necessary.
10. Do not claim that you searched documents or the web unless
    the corresponding tool was actually executed.
11. Avoid repeating searches that have already returned sufficient
    information.
12. After receiving useful tool results, provide an answer
    instead of unnecessarily requesting more tool calls.

DOCUMENT GROUNDING AND CONFLICT DETECTION RULES:

13. When multiple retrieved documents contain information relevant
    to the question, compare their statements before answering.

14. If two or more documents provide different values, dates,
    requirements, or policies that may conflict:

    - Explicitly identify each conflicting statement.
    - Identify the filename of each supporting document.
    - Include the page number when available.
    - Do not silently select one value over another.

15. Never invent explanations to reconcile conflicting information.

16. If documents contain effective dates, version numbers, or
    approval information, consider these when evaluating which
    document is authoritative.

17. Prefer one document over another only when its authority,
    effective date, or approval status is explicitly established.

18. If the available documents do not establish which information
    is authoritative, clearly explain the conflict and recommend
    contacting the appropriate department.

19. Never fabricate facts, policy interpretations, document names,
    effective dates, or explanations.

20. If retrieved documents do not contain sufficient evidence,
    acknowledge that the available information is insufficient.

21. If multiple documents agree on an answer, provide the answer
    directly without unnecessarily reporting a conflict.

Conversation history:
{context}

Known facts:
{facts}
"""

    # ==================================================
    # TOOL EXECUTION
    # ==================================================

    async def _execute_single_tool(
        self,
        user_id,
        tool_registry,
        tool_call
    ):

        tool_name = tool_call.function.name

        try:

            args = json.loads(
                tool_call.function.arguments
            )

            if not isinstance(args, dict):

                raise ValueError(
                    "Tool arguments must be a JSON object."
                )

            tool_fn = tool_registry.get(tool_name)

            if tool_fn is None:

                raise ValueError(
                    f"Unknown tool: {tool_name}"
                )

            # Never trust a user ID supplied by the LLM.
            args.pop("user_id", None)

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

            return str(result)

        except Exception as exc:

            logger.exception(
                "Tool execution failed: %s",
                tool_name
            )

            return f"Tool execution failed: {str(exc)}"

    # ==================================================
    # TOOL ORCHESTRATION
    # ==================================================

    async def _run_tool_orchestration(
        self,
        user_id,
        session_id,
        user_message,
        emit_status=None
    ):

        if emit_status is None:
            emit_status = lambda message: None

        system_prompt = self._build_system_prompt(
            user_id,
            session_id
        )

        messages = self.sessions[user_id][session_id]["messages"]

        llm_messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            *messages
        ]

        tool_registry = {
            **TOOLS,
            "retrieve_documents": self.retrieve_documents
        }

        executed_tools = []

        # ------------------------------------------
        # AUTOMATIC DOCUMENT RETRIEVAL
        # ------------------------------------------

        request_tool_schemas = TOOL_SCHEMAS

        if self._should_retrieve_documents(user_message):

            emit_status(
                "Identifying relevant company documents..."
            )

            emit_status(
                "Searching uploaded documents..."
            )

            document_result = await asyncio.to_thread(
                self.retrieve_documents,
                user_id,
                user_message,
                5
            )

            executed_tools.append({
                "name": "retrieve_documents",
                "result": document_result
            })

            llm_messages.append({
                "role": "system",
                "content": (
                    "DOCUMENT RETRIEVAL RESULTS:\n\n"
                    f"{document_result}\n\n"
                    "Use these retrieved documents as evidence "
                    "when answering the user's question. "
                    "Do not invent information that is not "
                    "supported by the retrieved content."
                )
            })

            # Prevent duplicate retrieval.
            request_tool_schemas = [
                schema
                for schema in TOOL_SCHEMAS
                if schema["function"]["name"] != "retrieve_documents"
            ]

        emit_status("Analyzing your request...")

        for iteration in range(self.max_tool_iterations):

            logger.info(
                "Agent tool iteration: %s",
                iteration + 1
            )

            response = await self.llm.chat.completions.create(
                model=self.llm.model_name,
                messages=llm_messages,
                tools=request_tool_schemas,
                tool_choice="auto",
                parallel_tool_calls=False,
                temperature=0
            )

            assistant_message = response.choices[0].message

            tool_calls = assistant_message.tool_calls

            if not tool_calls:

                logger.info("No additional tools requested.")

                return {
                    "direct_response": assistant_message.content or "",
                    "tool_results": executed_tools,
                    "conversation_messages": llm_messages
                }

            logger.info(
                "LLM selected %s tool(s)",
                len(tool_calls)
            )

            emit_status(
                f"Agent selected {len(tool_calls)} tool(s)."
            )

            llm_messages.append(
                assistant_message.model_dump(
                    exclude_none=True
                )
            )

            for tool_call in tool_calls:

                tool_name = tool_call.function.name

                emit_status(
                    self._get_tool_status(tool_name)
                )

                tool_result = await self._execute_single_tool(
                    user_id,
                    tool_registry,
                    tool_call
                )

                executed_tools.append({
                    "name": tool_name,
                    "result": tool_result
                })

                emit_status(
                    f"{self._get_tool_display_name(tool_name)} completed."
                )

                llm_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": tool_result
                })

        logger.warning(
            "Maximum tool iterations reached."
        )

        emit_status(
            "Preparing the final response..."
        )

        final_messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            *messages
        ]

        if executed_tools:

            tool_summary = "\n\n".join(
                [
                    f"Tool: {item['name']}\n"
                    f"Result: {item['result']}"
                    for item in executed_tools
                ]
            )

            final_messages.append({
                "role": "system",
                "content": (
                    "The following tool results were collected. "
                    "Use them as evidence when answering the user. "
                    "Do not invent information beyond these results.\n\n"
                    f"{tool_summary}"
                )
            })

        return {
            "direct_response": None,
            "tool_results": executed_tools,
            "conversation_messages": final_messages
        }

    # ==================================================
    # FINAL RESPONSE MESSAGE BUILDER
    # ==================================================

    def _build_final_messages(self, orchestration_result):

        return orchestration_result.get(
            "conversation_messages",
            []
        )

    # ==================================================
    # STRUCTURED STREAMING EVENTS
    # ==================================================

    async def stream_events(
        self,
        user_id,
        session_id,
        user_message
    ):

        self._ensure_session(user_id, session_id)

        full_reply = ""

        def emit_status(message):

            return self._emit_event(
                "status",
                {
                    "message": message
                }
            )

        try:

            # ------------------------------------------
            # SAVE USER MESSAGE
            # ------------------------------------------

            self.add_to_session(
                user_id,
                session_id,
                "user",
                user_message
            )

            self.memory.add_short_term(
                user_id,
                f"User: {user_message}"
            )

            if self.sessions[user_id][session_id]["title"] == "New Chat":

                self.sessions[user_id][session_id]["title"] = (
                    self.generate_title(user_message)
                )

            yield emit_status(
                "Analyzing your request..."
            )

            # ------------------------------------------
            # DOCUMENT INGESTION
            # ------------------------------------------

            normalized_message = user_message.strip()

            file_path = self._parse_ingestion_command(
                normalized_message
            )

            is_ingestion_command = (
                normalized_message.lower().strip() == "ingest"
                or file_path is not None
            )

            if is_ingestion_command:

                yield emit_status(
                    "Preparing document ingestion..."
                )

                if file_path:

                    yield emit_status(
                        "Loading and indexing your document..."
                    )

                full_reply = await asyncio.to_thread(
                    self._ingest_document,
                    user_id,
                    file_path
                )

                self.add_to_session(
                    user_id,
                    session_id,
                    "assistant",
                    full_reply
                )

                self.memory.add_short_term(
                    user_id,
                    f"Assistant: {full_reply}"
                )

                yield self._emit_event(
                    "token",
                    {
                        "content": full_reply
                    }
                )

                yield self._emit_event(
                    "done",
                    {
                        "message": "Document ingestion completed."
                    }
                )

                return

            # ------------------------------------------
            # LONG-TERM MEMORY EXTRACTION
            # ------------------------------------------

            fact_reply = self.extract_fact(
                user_id,
                user_message
            )

            if fact_reply:

                full_reply = fact_reply

                self.add_to_session(
                    user_id,
                    session_id,
                    "assistant",
                    full_reply
                )

                self.memory.add_short_term(
                    user_id,
                    f"Assistant: {full_reply}"
                )

                yield self._emit_event(
                    "token",
                    {
                        "content": full_reply
                    }
                )

                yield self._emit_event(
                    "done",
                    {
                        "message": "Response completed."
                    }
                )

                return

            # ------------------------------------------
            # TOOL ORCHESTRATION
            # ------------------------------------------

            system_prompt = self._build_system_prompt(
                user_id,
                session_id
            )

            messages = self.sessions[user_id][session_id]["messages"]

            llm_messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                *messages
            ]

            tool_registry = {
                **TOOLS,
                "retrieve_documents": self.retrieve_documents
            }

            executed_tools = []

            direct_response = None

            # ------------------------------------------
            # AUTOMATIC DOCUMENT RETRIEVAL
            # ------------------------------------------

            request_tool_schemas = TOOL_SCHEMAS

            if self._should_retrieve_documents(user_message):

                yield emit_status(
                    "Identifying relevant company documents..."
                )

                yield emit_status(
                    "Searching uploaded documents..."
                )

                document_result = await asyncio.to_thread(
                    self.retrieve_documents,
                    user_id,
                    user_message,
                    5
                )

                executed_tools.append({
                    "name": "retrieve_documents",
                    "result": document_result
                })

                llm_messages.append({
                    "role": "system",
                    "content": (
                        "DOCUMENT RETRIEVAL RESULTS:\n\n"
                        f"{document_result}\n\n"
                        "Use these retrieved documents as evidence "
                        "when answering the user's question. "
                        "Do not invent information that is not "
                        "supported by the retrieved content."
                    )
                })

                # Prevent the LLM from retrieving the same documents again.
                request_tool_schemas = [
                    schema
                    for schema in TOOL_SCHEMAS
                    if schema["function"]["name"] != "retrieve_documents"
                ]

            # ------------------------------------------
            # TOOL SELECTION LOOP
            # ------------------------------------------

            for iteration in range(self.max_tool_iterations):

                if iteration == 0:

                    yield emit_status(
                        "Deciding whether tools are needed..."
                    )

                else:

                    yield emit_status(
                        f"Reviewing tool results (step {iteration + 1})..."
                    )

                response = await self.llm.chat.completions.create(
                    model=self.llm.model_name,
                    messages=llm_messages,
                    tools=request_tool_schemas,
                    tool_choice="auto",
                    parallel_tool_calls=False,
                    temperature=0
                )

                assistant_message = response.choices[0].message

                tool_calls = assistant_message.tool_calls

                if not tool_calls:

                    direct_response = (
                        assistant_message.content or ""
                    )

                    break

                yield emit_status(
                    f"Agent selected {len(tool_calls)} tool(s)."
                )

                llm_messages.append(
                    assistant_message.model_dump(
                        exclude_none=True
                    )
                )

                for tool_call in tool_calls:

                    tool_name = tool_call.function.name

                    yield emit_status(
                        self._get_tool_status(tool_name)
                    )

                    tool_result = await self._execute_single_tool(
                        user_id,
                        tool_registry,
                        tool_call
                    )

                    executed_tools.append({
                        "name": tool_name,
                        "result": tool_result
                    })

                    yield emit_status(
                        f"{self._get_tool_display_name(tool_name)} completed."
                    )

                    llm_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": tool_result
                    })

            else:

                direct_response = None

            # ------------------------------------------
            # FINAL RESPONSE PREPARATION
            # ------------------------------------------

            if direct_response is not None:

                full_reply = direct_response

                if full_reply.strip():

                    yield self._emit_event(
                        "token",
                        {
                            "content": full_reply
                        }
                    )

            else:

                yield emit_status(
                    "Preparing the final response..."
                )

                final_messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    *messages
                ]

                if executed_tools:

                    tool_summary = "\n\n".join(
                        [
                            f"Tool: {item['name']}\n"
                            f"Result: {item['result']}"
                            for item in executed_tools
                        ]
                    )

                    final_messages.append({
                        "role": "system",
                        "content": (
                            "The following tool results were collected. "
                            "Use them as evidence when answering the user. "
                            "Do not invent information beyond these results. "
                            "Do not request additional tools. "
                            "Provide the best complete answer using the "
                            "available results.\n\n"
                            f"{tool_summary}"
                        )
                    })

                # --------------------------------------
                # STREAM FINAL LLM RESPONSE
                # --------------------------------------

                yield emit_status(
                    "Generating your response..."
                )

                response_stream = (
                    await self.llm.chat.completions.create(
                        model=self.llm.model_name,
                        messages=final_messages,
                        temperature=0,
                        stream=True
                    )
                )

                async for chunk in response_stream:

                    try:

                        token = ""

                        if hasattr(chunk, "choices"):

                            if not chunk.choices:
                                continue

                            delta = chunk.choices[0].delta

                            if delta and delta.content:
                                token = delta.content

                        elif isinstance(chunk, dict):

                            choices = chunk.get("choices", [])

                            if choices:

                                delta = choices[0].get(
                                    "delta",
                                    {}
                                )

                                if isinstance(delta, dict):

                                    token = (
                                        delta.get("content", "") or ""
                                    )

                        if token:

                            full_reply += token

                            yield self._emit_event(
                                "token",
                                {
                                    "content": token
                                }
                            )

                    except Exception:

                        logger.exception(
                            "Failed to process streaming chunk."
                        )

                        continue

            # ------------------------------------------
            # EMPTY RESPONSE HANDLING
            # ------------------------------------------

            if not full_reply.strip():

                full_reply = (
                    "Sorry, I couldn't generate a response."
                )

                yield self._emit_event(
                    "token",
                    {
                        "content": full_reply
                    }
                )

            # ------------------------------------------
            # SAVE ASSISTANT RESPONSE
            # ------------------------------------------

            self.add_to_session(
                user_id,
                session_id,
                "assistant",
                full_reply
            )

            self.memory.add_short_term(
                user_id,
                f"Assistant: {full_reply}"
            )

            logger.info(
                "Structured streaming completed successfully."
            )

            yield self._emit_event(
                "done",
                {
                    "message": "Response completed."
                }
            )

        except Exception as exc:

            logger.exception(
                "Agent.stream_events failed: %s",
                exc
            )

            fallback = (
                "Sorry, the assistant is temporarily unavailable."
            )

            yield self._emit_event(
                "error",
                {
                    "message": str(exc)
                }
            )

            yield self._emit_event(
                "token",
                {
                    "content": fallback
                }
            )

            yield self._emit_event(
                "done",
                {
                    "message": "Response completed with an error."
                }
            )

    # ==================================================
    # BACKWARD-COMPATIBLE PLAIN TEXT STREAMING
    # ==================================================

    async def stream(self, user_id, session_id, user_message):

        async for event in self.stream_events(
            user_id,
            session_id,
            user_message
        ):

            if event.get("event") == "token":

                content = event.get(
                    "data",
                    {}
                ).get("content", "")

                if content:
                    yield content

    # ==================================================
    # TOOL EXECUTION (NON-STREAMING)
    # ==================================================

    async def _execute_tool_calls(
        self,
        user_id,
        session_id,
        user_message
    ):

        tool_registry = {
            **TOOLS,
            "retrieve_documents": self.retrieve_documents
        }

        messages = self.sessions[user_id][session_id]["messages"]

        system_prompt = self._build_system_prompt(
            user_id,
            session_id
        )

        llm_messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            *messages
        ]

        executed_tools = []

        # ------------------------------------------
        # AUTOMATIC DOCUMENT RETRIEVAL
        # ------------------------------------------

        request_tool_schemas = TOOL_SCHEMAS

        if self._should_retrieve_documents(user_message):

            logger.info(
                "Automatically retrieving documents for user %s",
                user_id
            )

            document_result = await asyncio.to_thread(
                self.retrieve_documents,
                user_id,
                user_message,
                5
            )

            executed_tools.append({
                "name": "retrieve_documents",
                "result": document_result
            })

            llm_messages.append({
                "role": "system",
                "content": (
                    "DOCUMENT RETRIEVAL RESULTS:\n\n"
                    f"{document_result}\n\n"
                    "Use these retrieved documents as evidence "
                    "when answering the user's question. "
                    "Do not invent information that is not "
                    "supported by the retrieved content."
                )
            })

            # Prevent duplicate retrieval.
            request_tool_schemas = [
                schema
                for schema in TOOL_SCHEMAS
                if schema["function"]["name"] != "retrieve_documents"
            ]

        # ------------------------------------------
        # TOOL SELECTION LOOP
        # ------------------------------------------

        for iteration in range(self.max_tool_iterations):

            response = await self.llm.chat.completions.create(
                model=self.llm.model_name,
                messages=llm_messages,
                tools=request_tool_schemas,
                tool_choice="auto",
                parallel_tool_calls=False,
                temperature=0
            )

            assistant_message = response.choices[0].message

            tool_calls = assistant_message.tool_calls

            if not tool_calls:

                return assistant_message.content or ""

            logger.info(
                "LLM selected %s tool(s)",
                len(tool_calls)
            )

            llm_messages.append(
                assistant_message.model_dump(
                    exclude_none=True
                )
            )

            for tool_call in tool_calls:

                tool_name = tool_call.function.name

                tool_result = await self._execute_single_tool(
                    user_id,
                    tool_registry,
                    tool_call
                )

                executed_tools.append({
                    "name": tool_name,
                    "result": tool_result
                })

                llm_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": tool_result
                })

        # ------------------------------------------
        # MAXIMUM TOOL ITERATIONS REACHED
        # ------------------------------------------

        logger.warning(
            "Maximum tool iterations reached."
        )

        final_messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            *messages
        ]

        if executed_tools:

            tool_summary = "\n\n".join(
                [
                    f"Tool: {item['name']}\n"
                    f"Result: {item['result']}"
                    for item in executed_tools
                ]
            )

            final_messages.append({
                "role": "system",
                "content": (
                    "Use these tool results to answer the user. "
                    "Do not request additional tools. "
                    "Do not invent information beyond the results.\n\n"
                    f"{tool_summary}"
                )
            })

        response = await self.llm.chat.completions.create(
            model=self.llm.model_name,
            messages=final_messages,
            temperature=0
        )

        return response.choices[0].message.content or ""

    # ==================================================
    # NON-STREAMING EXECUTION
    # ==================================================

    async def run(self, user_id, session_id, user_message):

        self._ensure_session(user_id, session_id)

        try:

            self.add_to_session(
                user_id,
                session_id,
                "user",
                user_message
            )

            if self.sessions[user_id][session_id]["title"] == "New Chat":

                self.sessions[user_id][session_id]["title"] = (
                    self.generate_title(user_message)
                )

            self.memory.add_short_term(
                user_id,
                f"User: {user_message}"
            )

            # ------------------------------------------
            # DOCUMENT INGESTION
            # ------------------------------------------

            normalized_message = user_message.strip()

            file_path = self._parse_ingestion_command(
                normalized_message
            )

            is_ingestion_command = (
                normalized_message.lower().strip() == "ingest"
                or file_path is not None
            )

            if is_ingestion_command:

                logger.info(
                    "DOCUMENT INGESTION COMMAND DETECTED"
                )

                assistant_reply = await asyncio.to_thread(
                    self._ingest_document,
                    user_id,
                    file_path
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

                return assistant_reply

            # ------------------------------------------
            # FACT EXTRACTION
            # ------------------------------------------

            fact_reply = self.extract_fact(
                user_id,
                user_message
            )

            if fact_reply:

                self.add_to_session(
                    user_id,
                    session_id,
                    "assistant",
                    fact_reply
                )

                self.memory.add_short_term(
                    user_id,
                    f"Assistant: {fact_reply}"
                )

                return fact_reply

            # ------------------------------------------
            # LLM TOOL ORCHESTRATION
            # ------------------------------------------

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

            # ------------------------------------------
            # SAVE ASSISTANT RESPONSE
            # ------------------------------------------

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

            return assistant_reply

        except Exception as exc:

            logger.exception(
                "Agent.run unexpected error: %s",
                exc
            )

            fallback = (
                "Sorry, the assistant is temporarily unavailable."
            )

            return fallback