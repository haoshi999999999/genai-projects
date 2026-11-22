import pytest

from unittest.mock import AsyncMock, MagicMock
from types import SimpleNamespace

from agent.agent import Agent


# ==================================================
# MOCK LLM RESPONSE HELPERS
# ==================================================

def create_llm_response(content=None, tool_calls=None):
    """
    Create a mock response that resembles the response
    returned by the Groq chat completions API.
    """

    assistant_message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls
    )

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=assistant_message
            )
        ]
    )


def create_tool_call(tool_name, arguments, call_id="call_123"):
    """
    Create a mock tool call resembling the structure
    returned by the LLM.
    """

    function = SimpleNamespace(
        name=tool_name,
        arguments=arguments
    )

    tool_call = SimpleNamespace(
        id=call_id,
        function=function
    )

    return tool_call


def create_tool_call_response(tool_call):
    """
    Create an assistant response containing a tool call.

    model_dump() is needed because your Agent uses it
    to append the assistant tool-call message.
    """

    assistant_message = SimpleNamespace(
        content=None,
        tool_calls=[tool_call],
        model_dump=lambda exclude_none=True: {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
            ]
        }
    )

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=assistant_message
            )
        ]
    )


# ==================================================
# AGENT FIXTURE
# ==================================================

@pytest.fixture
def agent():
    """
    Create an Agent without running its real constructor.

    This avoids initializing:
    - HuggingFace embeddings
    - FAISS document storage
    - Real memory systems
    - External services
    """

    test_agent = Agent.__new__(Agent)

    # Mock LLM
    test_agent.llm = MagicMock()
    test_agent.llm.model_name = "mock-model"
    test_agent.llm.chat.completions.create = AsyncMock()

    # Mock memory systems
    test_agent.memory = MagicMock()
    test_agent.semantic = MagicMock()

    # Mock document ingestion
    test_agent.docs = MagicMock()

    # Session storage
    test_agent.history = {}
    test_agent.sessions = {}

    # Personas
    test_agent.personas = {
        "default": "You are a helpful AI assistant.",
        "friendly": "You respond warmly.",
        "technical": "You respond technically.",
        "coach": "You respond like a coach.",
        "concise": "You respond concisely."
    }

    # Tool iteration limit
    test_agent.max_tool_iterations = 3

    return test_agent


# ==================================================
# TEST 1: DIRECT LLM RESPONSE
# ==================================================

@pytest.mark.asyncio
async def test_run_direct_response(agent):
    """
    Verify that the Agent can answer a question
    without executing any tools.
    """

    # Arrange
    agent.llm.chat.completions.create.return_value = (
        create_llm_response(
            content="RAG stands for Retrieval-Augmented Generation.",
            tool_calls=None
        )
    )

    # Act
    result = await agent.run(
        user_id="user_1",
        session_id="session_1",
        user_message="What is RAG?"
    )

    # Assert
    assert result == (
        "RAG stands for Retrieval-Augmented Generation."
    )

    # LLM should be called once.
    assert agent.llm.chat.completions.create.await_count == 1

    # No tool should have been executed.
    agent.memory.add_short_term.assert_any_call(
        "user_1",
        "User: What is RAG?"
    )

    agent.memory.add_short_term.assert_any_call(
        "user_1",
        "Assistant: RAG stands for Retrieval-Augmented Generation."
    )

    # Both user and assistant messages should be saved.
    messages = agent.sessions["user_1"]["session_1"]["messages"]

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


# ==================================================
# TEST 2: LLM SELECTS AND EXECUTES A TOOL
# ==================================================

@pytest.mark.asyncio
async def test_run_executes_add_numbers_tool(agent):
    """
    Verify that the LLM can select add_numbers,
    the Agent executes it, and the LLM generates
    a final response using the tool result.
    """

    # Arrange
    tool_call = create_tool_call(
        tool_name="add_numbers",
        arguments='{"a": 10, "b": 20}'
    )

    tool_response = create_tool_call_response(
        tool_call
    )

    final_response = create_llm_response(
        content="The result is 30.",
        tool_calls=None
    )

    agent.llm.chat.completions.create.side_effect = [
        tool_response,
        final_response
    ]

    # Act
    result = await agent.run(
        user_id="user_1",
        session_id="session_1",
        user_message="What is 10 plus 20?"
    )

    # Assert
    assert result == "The result is 30."

    # First LLM call selects the tool.
    # Second LLM call generates the final answer.
    assert agent.llm.chat.completions.create.await_count == 2

    # Verify the second LLM request contains
    # the tool execution result.
    second_call = agent.llm.chat.completions.create.await_args_list[1]

    messages = second_call.kwargs["messages"]

    tool_messages = [
        message
        for message in messages
        if message["role"] == "tool"
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0]["name"] == "add_numbers"
    assert tool_messages[0]["content"] == "30"

    # Verify assistant response was saved.
    saved_messages = agent.sessions["user_1"]["session_1"]["messages"]

    assert saved_messages[-1]["content"] == "The result is 30."


# ==================================================
# TEST 3: AUTOMATIC DOCUMENT RETRIEVAL
# ==================================================

@pytest.mark.asyncio
async def test_run_automatically_retrieves_documents(agent):
    """
    Verify that a policy-related question triggers
    automatic document retrieval before the LLM responds.
    """

    # Arrange
    agent.retrieve_documents = MagicMock(
        return_value=(
            "Document: Employee Handbook\n"
            "Vacation allowance: 15 days."
        )
    )

    agent.llm.chat.completions.create.return_value = (
        create_llm_response(
            content="The handbook provides 15 vacation days.",
            tool_calls=None
        )
    )

    # Act
    result = await agent.run(
        user_id="user_1",
        session_id="session_1",
        user_message="How many vacation days do employees get?"
    )

    # Assert
    assert result == (
        "The handbook provides 15 vacation days."
    )

    # Automatic retrieval should have happened.
    agent.retrieve_documents.assert_called_once()

    # The retrieval result should be included
    # in the LLM's first request.
    first_call = agent.llm.chat.completions.create.await_args_list[0]

    messages = first_call.kwargs["messages"]

    document_messages = [
        message
        for message in messages
        if "DOCUMENT RETRIEVAL RESULTS" in message.get("content", "")
    ]

    assert len(document_messages) == 1

    assert "15 days" in document_messages[0]["content"]

    # Automatic retrieval should remove the
    # retrieve_documents tool from available schemas.
    tool_schemas = first_call.kwargs["tools"]

    available_tool_names = [
        schema["function"]["name"]
        for schema in tool_schemas
    ]

    assert "retrieve_documents" not in available_tool_names


# ==================================================
# TEST 4: MAXIMUM TOOL ITERATION LIMIT
# ==================================================

@pytest.mark.asyncio
async def test_run_respects_max_tool_iterations(agent):
    """
    Verify that the Agent stops selecting tools
    after reaching max_tool_iterations.

    A final LLM call should then generate the answer.
    """

    # Arrange
    tool_call = create_tool_call(
        tool_name="echo",
        arguments='{"text": "test"}'
    )

    tool_response = create_tool_call_response(
        tool_call
    )

    # The LLM requests another tool on every
    # orchestration iteration.
    agent.llm.chat.completions.create.side_effect = [
        tool_response,
        tool_response,
        tool_response,
        create_llm_response(
            content="Here is the final answer.",
            tool_calls=None
        )
    ]

    # Act
    result = await agent.run(
        user_id="user_1",
        session_id="session_1",
        user_message="Run the echo tool."
    )

    # Assert
    assert result == "Here is the final answer."

    # Three tool-selection calls plus one final response call.
    assert agent.llm.chat.completions.create.await_count == 4


# ==================================================
# TEST 5: STREAMING DIRECT RESPONSE
# ==================================================

@pytest.mark.asyncio
async def test_stream_events_direct_response(agent):
    """
    Verify that stream_events emits:
    - status event
    - token event
    - done event

    This test does not require a real streaming LLM response
    because the mocked LLM returns a direct response.
    """

    # Arrange
    agent.llm.chat.completions.create.return_value = (
        create_llm_response(
            content="Hello! How can I help?",
            tool_calls=None
        )
    )

    # Act
    events = [
        event
        async for event in agent.stream_events(
            user_id="user_1",
            session_id="session_1",
            user_message="Hello"
        )
    ]

    # Assert
    event_types = [
        event["event"]
        for event in events
    ]

    assert "status" in event_types
    assert "token" in event_types
    assert "done" in event_types

    # Check token content.
    token_events = [
        event
        for event in events
        if event["event"] == "token"
    ]

    assert len(token_events) == 1

    assert token_events[0]["data"]["content"] == (
        "Hello! How can I help?"
    )

    # Verify the final event.
    assert events[-1]["event"] == "done"

    # Verify session persistence.
    messages = agent.sessions["user_1"]["session_1"]["messages"]

    assert len(messages) == 2
    assert messages[0]["content"] == "Hello"
    assert messages[1]["content"] == "Hello! How can I help?"


# ==================================================
# TEST 6: TOOL EXECUTION ERROR HANDLING
# ==================================================

@pytest.mark.asyncio
async def test_execute_single_tool_handles_invalid_arguments(agent):
    """
    Verify that malformed tool arguments do not crash
    the Agent and that internal exception details
    are not exposed to the LLM.
    """

    # Arrange
    tool_call = create_tool_call(
        tool_name="add_numbers",
        arguments="this is not valid JSON"
    )

    # Act
    result = await agent._execute_single_tool(
        user_id="user_1",
        tool_registry={
            "add_numbers": lambda a, b: a + b
        },
        tool_call=tool_call
    )

    # Assert
    assert result == (
        "Tool 'add_numbers' failed. "
        "The tool could not complete the requested operation."
    )