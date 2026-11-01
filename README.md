# GenAI Core --- AI Agent Application

A personal project exploring a modular, LLM-powered conversational AI
application with agent orchestration, tool execution, user
authentication, PostgreSQL persistence, and multi-layer conversational
memory.

> **Project status:** Personal learning and development project. The
> implementation is evolving; see [Current limitations and future
> improvements](#current-limitations-and-future-improvements).

## Overview

GenAI Core brings together: - An LLM-powered agent workflow. - Agent
routing and tool execution. - Configurable conversational personas. -
Short-term and long-term conversation memory. - Semantic memory
retrieval using embeddings and FAISS. - Document ingestion and retrieval
functionality. - A FastAPI backend and PostgreSQL data models for users,
sessions, and messages.

## Architecture

The following is a high-level conceptual view of the application based
on the modules currently shared. Confirm exact request paths against the
implementation as it evolves.

``` mermaid
flowchart TD
    U[User] --> UI[User Interface]
    UI --> API[FastAPI Backend]
    API --> AUTH[Authentication]
    AUTH --> DB[(PostgreSQL)]
    API --> AGENT[AI Agent]
    AGENT --> ROUTER[Agent Router]
    ROUTER --> LLM[LLM Integration]
    ROUTER --> TOOLS[Tools]
    TOOLS --> MATH[Math Utilities]
    TOOLS --> SEARCH[Search]
    TOOLS --> UTIL[Other Utilities]
    AGENT --> MEM[Memory Components]
    MEM --> STM[Short-Term Memory]
    MEM --> LTM[Long-Term Memory]
    MEM --> SEM[Semantic Memory]
    SEM --> EMB[Hugging Face Embeddings]
    EMB --> FAISS[(FAISS Vector Store)]
    DB --> SESS[Users, Sessions and Messages]
```

## Main components

### 1. AI agent and orchestration

The `agent/` package contains the agent workflow and routing-related
modules. It coordinates language-model interactions and supports routing
between conversational responses and tool execution. A separate
tool-chain module supports workflows involving tools.

### 2. Tool integration

The `tools/` package contains utilities that the agent can invoke,
including mathematical operations, search functionality, and general
utility functions. Available tools depend on the agent's routing logic
and configuration.

### 3. Conversational memory

The `memory/` package contains several memory approaches:

  -----------------------------------------------------------------------
  Component                           Purpose
  ----------------------------------- -----------------------------------
  Short-term memory                   Retains a limited window of recent
                                      conversation entries.

  Long-term memory                    Stores facts explicitly provided to
                                      be remembered.

  Semantic memory                     Uses text embeddings and FAISS
                                      similarity search to retrieve
                                      related memory content.

  Document ingestion                  Provides document-loading and
                                      retrieval functionality for
                                      document-based interactions.
  -----------------------------------------------------------------------

The semantic-memory implementation uses the
`sentence-transformers/all-MiniLM-L6-v2` embedding model through Hugging
Face integrations and LangChain's FAISS vector store.

### 4. Authentication and data persistence

The application includes authentication logic and a PostgreSQL-backed
data model. The SQLAlchemy models define:

-   **Users:** account identity and a stored password-hash field.
-   **Sessions:** conversation sessions associated with users.
-   **Messages:** user and assistant messages associated with sessions,
    including creation timestamps.

The database layer uses SQLAlchemy's asynchronous engine/session
facilities. Alembic is included for database schema migrations.

**Security note:** This is a personal project, not a security-audited
authentication service. Before using it with real users or publishing it
as production-ready, password storage, secret management, token
handling, session authorization, and user-data isolation need to be
reviewed and hardened. Do not use real credentials or personal
conversation data in the repository.

### 5. Backend and interface

-   `server.py` contains the FastAPI application/backend entry point.
-   `ui.py` contains the user interface.
-   `db/` contains database setup and ORM models.
-   `alembic/` contains migration configuration and migration scripts.

## Technology stack

Technologies represented in the project source include:

-   **Language:** Python
-   **LLM / agent framework:** LangChain
-   **LLM provider integration:** Groq
-   **Backend:** FastAPI
-   **Database:** PostgreSQL
-   **ORM / asynchronous database access:** SQLAlchemy
-   **Database migrations:** Alembic
-   **Embeddings:** Hugging Face Sentence Transformers
-   **Vector store:** FAISS
-   **Configuration:** environment variables with `python-dotenv`

## Repository structure

``` text
genai-projects/
├── genai_core/
│   ├── agent/
│   │   ├── agent.py
│   │   ├── router.py
│   │   └── tool_chain.py
│   ├── alembic/
│   │   └── versions/
│   ├── db/
│   │   ├── database.py
│   │   └── models.py
│   ├── memory/
│   │   ├── document_ingestion.py
│   │   ├── memory.py
│   │   ├── multi_user_memory.py
│   │   └── semantic_memory.py
│   ├── models/
│   ├── tools/
│   ├── auth.py
│   ├── server.py
│   └── ui.py
├── .gitignore
└── README.md
```

## Local setup

The steps below are a general setup outline. Adjust commands and
configuration to match the current application entry points and
dependencies.

### 1. Clone the repository

``` bash
git clone https://github.com/haoshi999999999/genai-projects.git
cd genai-projects
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

``` bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If `requirements.txt` is present:

``` bash
pip install -r requirements.txt
```

If it has not yet been added, create and maintain one from the working
environment before asking others to run the application. Do not commit
the virtual environment.

### 4. Configure environment variables

Create a local `.env` file and provide the configuration required by the
application. For example:

``` dotenv
DATABASE_URL=postgresql+asyncpg://<username>:<password>@localhost:5432/<database_name>
GROQ_API_KEY=<your_groq_api_key>
SECRET_KEY=<a_long_random_secret>
```

These are illustrative variable names. Confirm the exact names used by
the current code before running the application.

-   Do not commit `.env`.
-   Do not put API keys, database passwords, signing keys, or real user
    data in source files.
-   Use `.env.example` to document required variable names with
    placeholder values only.

### 5. Prepare PostgreSQL and migrations

Create a PostgreSQL database and configure its connection string. Apply
the Alembic migrations using the project's migration configuration.
Confirm the exact command and working directory against the current
`alembic.ini` and migration environment.

### 6. Run the application

The backend entry point is `genai_core/server.py`. Confirm the
module/import setup and FastAPI application object name before choosing
the Uvicorn command.

For example, if the application object is named `app` and the package is
importable from the repository root:

``` bash
uvicorn genai_core.server:app --reload
```

## Current limitations and future improvements

This project is under personal development. Areas for further work
include:

-   Replace fast, unsalted SHA-256 password hashing with a dedicated
    password-hashing algorithm such as Argon2id or an appropriately
    configured PBKDF2 implementation.
-   Load signing keys and service credentials from environment variables
    or a secrets manager.
-   Use a maintained JWT library and enforce required claims, including
    token expiration.
-   Enforce ownership checks so users can access only their own sessions
    and messages.
-   Add user-level isolation to semantic memory retrieval. The current
    semantic-memory example should not be treated as safely isolated
    across users until this is implemented and tested.
-   Add automated tests for authentication, authorization, agent
    routing, memory retrieval, and API endpoints.
-   Improve persistence, concurrency handling, and lifecycle management
    for memory stores.
-   Document and test the complete setup and execution workflow.

## Disclaimer

This repository is a personal software development project intended for
learning and demonstration. It is not presented as a production-ready or
security-audited service. Do not deploy it with real user credentials or
sensitive data without further security, privacy, reliability, and
operational work.

## Author

Developed independently as a personal project exploring LLM-powered
applications, agent workflows, backend engineering, and conversational
memory.

GitHub: https://github.com/haoshi999999999
