import os
import uuid
import hashlib
import traceback
import bcrypt
import json
import logging

import asyncio
import tempfile
from pathlib import Path

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import jwt
import secrets

from pwdlib import PasswordHash
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    UploadFile,
    File
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from groq import AsyncGroq, NotFoundError

from db.database import async_session
from db.models import User, Session, Message

from agent.agent import Agent


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

FALLBACK_GROQ_MODEL = os.getenv(
    "FALLBACK_GROQ_MODEL",
    None
)

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing. "
        "Add it to your .env file or environment."
    )


# --------------------------------------------------
# AUTHENTICATION CONFIGURATION
# --------------------------------------------------

SECRET_KEY = os.getenv("JWT_SECRET_KEY")

if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY is missing from .env")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

password_hash = PasswordHash.recommended()
security = HTTPBearer(auto_error=False)


# --------------------------------------------------
# DOCUMENT UPLOAD CONFIGURATION
# --------------------------------------------------

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

ALLOWED_DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".txt"
}


# --------------------------------------------------
# GROQ CLIENT WRAPPER
# --------------------------------------------------

class GroqLLM:

    def __init__(self, api_key: str, model_name: str):
        self.client = AsyncGroq(api_key=api_key)
        self.chat = self.client.chat
        self.model_name = model_name

    async def invoke(
        self,
        prompt: str,
        *,
        stream: bool = False
    ):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Answer clearly and concisely."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        try:
            response = await self.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                stream=False
            )

        except NotFoundError:
            logger.error("Model not found: %s", self.model_name)

            if not FALLBACK_GROQ_MODEL:
                raise

            logger.info(
                "Retrying with fallback model: %s",
                FALLBACK_GROQ_MODEL
            )

            self.model_name = FALLBACK_GROQ_MODEL

            response = await self.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                stream=False
            )

        content = response.choices[0].message.content

        logger.info("Groq model: %s", self.model_name)

        if not content or not content.strip():
            raise RuntimeError("Groq returned an empty response.")

        return content.strip()


# --------------------------------------------------
# INITIALIZE LLM AND AGENT
# --------------------------------------------------

llm = GroqLLM(
    api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL
)

agent = Agent(llm=llm)


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting GenAI Agent API.")
    yield
    logger.info("Shutting down GenAI Agent API.")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# --------------------------------------------------
# DATABASE HELPERS
# --------------------------------------------------

async def get_db():
    async with async_session() as db:
        yield db


# --------------------------------------------------
# PASSWORD HELPERS
# --------------------------------------------------

def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(
    password: str,
    hashed_password: str
) -> bool:

    # Argon2 password hashes
    if hashed_password.startswith("$argon2"):
        return password_hash.verify(
            password,
            hashed_password
        )

    # Bcrypt password hashes
    if hashed_password.startswith(
        ("$2a$", "$2b$", "$2y$")
    ):
        return bcrypt.checkpw(
            password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )

    # Legacy SHA-256 password hashes
    if len(hashed_password) == 64:
        legacy_hash = hashlib.sha256(
            password.encode()
        ).hexdigest()

        return secrets.compare_digest(
            legacy_hash,
            hashed_password
        )

    return False


# --------------------------------------------------
# JWT HELPERS
# --------------------------------------------------

def create_access_token(email: str):
    expire = datetime.now(
        timezone.utc
    ) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": email,
        "exp": expire
    }

    return jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM
    )


# --------------------------------------------------
# AUTHENTICATION DEPENDENCY
# --------------------------------------------------

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):

    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )

    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        email = payload.get("sub")

        if not email:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token"
            )

    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )

    result = await db.execute(
        select(User).where(User.email == email)
    )

    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=401,
            detail="User not found"
        )

    return user


# --------------------------------------------------
# SESSION OWNERSHIP VALIDATION
# --------------------------------------------------

async def find_owned_session(
    session_id: uuid.UUID,
    current_user: User,
    db: AsyncSession
):

    result = await db.execute(
        select(Session).where(
            Session.id == session_id,
            Session.user_email == current_user.email
        )
    )

    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )

    return session


# --------------------------------------------------
# INITIALIZE AGENT SESSION
# --------------------------------------------------

async def initialize_agent_session(
    session: Session,
    db: AsyncSession
):

    user_id = session.user_email
    session_id = str(session.id)

    if user_id not in agent.sessions:
        agent.sessions[user_id] = {}

    if session_id in agent.sessions[user_id]:
        agent.sessions[user_id][session_id]["persona"] = (
            session.persona
        )
        return user_id, session_id

    result = await db.execute(
        select(Message)
        .where(Message.session_id == session.id)
        .order_by(
            Message.created_at.asc(),
            Message.id.asc()
        )
    )

    stored_messages = result.scalars().all()

    agent.sessions[user_id][session_id] = {
        "title": session.title,
        "persona": session.persona,
        "messages": [
            {
                "role": message.role,
                "content": message.content
            }
            for message in stored_messages
        ]
    }

    return user_id, session_id


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class RegisterRequest(BaseModel):
    email: str
    password: str


class TestLLMRequest(BaseModel):
    prompt: str


class LoginRequest(BaseModel):
    email: str
    password: str


class CreateSessionRequest(BaseModel):
    persona: str = "default"
    title: str = "New Chat"


class MessageRequest(BaseModel):
    session_id: uuid.UUID
    content: str


class PersonaUpdate(BaseModel):
    session_id: uuid.UUID
    persona: str


class SessionRequest(BaseModel):
    session_id: uuid.UUID


# --------------------------------------------------
# AUTHENTICATION ENDPOINTS
# --------------------------------------------------

@app.post("/register")
async def register(
    req: RegisterRequest,
    db: AsyncSession = Depends(get_db)
):

    result = await db.execute(
        select(User).where(User.email == req.email)
    )

    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="User already exists"
        )

    user = User(
        email=req.email,
        password_hash=hash_password(req.password)
    )

    db.add(user)
    await db.commit()

    return {"status": "registered"}


@app.post("/login")
async def login(
    req: LoginRequest,
    db: AsyncSession = Depends(get_db)
):

    result = await db.execute(
        select(User).where(User.email == req.email)
    )

    user = result.scalar_one_or_none()

    if not user or not verify_password(
        req.password,
        user.password_hash
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    # Upgrade legacy hashes to Argon2.
    if not user.password_hash.startswith("$argon2"):
        user.password_hash = hash_password(req.password)
        await db.commit()

    access_token = create_access_token(user.email)

    return {
        "status": "logged_in",
        "access_token": access_token,
        "token_type": "bearer"
    }


# --------------------------------------------------
# SESSION ENDPOINTS
# --------------------------------------------------

@app.get("/session/list")
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Return only the authenticated user's sessions.

    Sessions are ordered by their most recent message activity,
    newest first. Sessions without messages appear after sessions
    that contain messages. The response includes last_activity so
    the UI can distinguish sessions with similar titles.
    """

    result = await db.execute(
        select(
            Session,
            func.max(Message.created_at).label("last_activity")
        )
        .outerjoin(
            Message,
            Message.session_id == Session.id
        )
        .where(
            Session.user_email == current_user.email
        )
        .group_by(Session.id)
        .order_by(
            func.max(Message.created_at).desc().nullslast(),
            Session.id.desc()
        )
    )

    session_rows = result.all()

    response = {}

    for session, last_activity in session_rows:
        response[str(session.id)] = {
            "title": session.title,
            "persona": session.persona,
            "last_activity": (
                last_activity.isoformat()
                if last_activity is not None
                else None
            )
        }

    return response


@app.post("/session/create")
async def create_session(
    req: CreateSessionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):

    if req.persona not in agent.personas:
        raise HTTPException(
            status_code=400,
            detail="Invalid persona"
        )

    new_session = Session(
        id=uuid.uuid4(),
        user_email=current_user.email,
        persona=req.persona,
        title=req.title
    )

    db.add(new_session)
    await db.commit()
    await db.refresh(new_session)

    return {
        "session_id": new_session.id
    }


@app.get("/session/{session_id}")
async def get_session(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):

    session = await find_owned_session(
        session_id,
        current_user,
        db
    )

    return {
        "id": session.id,
        "title": session.title,
        "persona": session.persona
    }


@app.post("/session/persona")
async def update_persona(
    req: PersonaUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):

    session = await find_owned_session(
        req.session_id,
        current_user,
        db
    )

    if req.persona not in agent.personas:
        raise HTTPException(
            status_code=400,
            detail="Invalid persona"
        )

    session.persona = req.persona
    await db.commit()

    user_id = session.user_email
    session_id = str(session.id)

    if (
        user_id in agent.sessions
        and session_id in agent.sessions[user_id]
    ):
        agent.sessions[user_id][session_id]["persona"] = (
            req.persona
        )

    return {"status": "updated"}


@app.post("/session/clear")
async def clear_session(
    req: SessionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):

    session = await find_owned_session(
        req.session_id,
        current_user,
        db
    )

    await db.execute(
        Message.__table__.delete().where(
            Message.session_id == session.id
        )
    )

    session.title = "New Chat"
    await db.commit()

    user_id = session.user_email
    agent_session_id = str(session.id)

    if (
        user_id in agent.sessions
        and agent_session_id in agent.sessions[user_id]
    ):
        agent.sessions[user_id][agent_session_id]["messages"] = []
        agent.sessions[user_id][agent_session_id]["title"] = "New Chat"

    return {
        "status": "cleared",
        "session_id": str(session.id),
        "title": "New Chat"
    }


# --------------------------------------------------
# CHAT ENDPOINTS
# --------------------------------------------------

@app.post("/message/send")
async def send_message(
    req: MessageRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):

    session = await find_owned_session(
        req.session_id,
        current_user,
        db
    )

    user_id, agent_session_id = await initialize_agent_session(
        session,
        db
    )

    # Persist user message once.
    user_msg = Message(
        id=uuid.uuid4(),
        session_id=session.id,
        role="user",
        content=req.content
    )

    db.add(user_msg)
    await db.commit()

    try:
        reply = await agent.run(
            user_id,
            agent_session_id,
            req.content
        )

        if not isinstance(reply, str) or not reply.strip():
            raise RuntimeError("Agent returned an empty response.")

    except Exception as exc:
        logger.exception("Agent.run failed: %s", exc)
        reply = "Sorry, the assistant is temporarily unavailable."

    # Persist assistant response once.
    assistant_msg = Message(
        id=uuid.uuid4(),
        session_id=session.id,
        role="assistant",
        content=reply
    )

    db.add(assistant_msg)
    await db.commit()

    session.title = (
        agent.sessions[user_id][agent_session_id]["title"]
    )

    await db.commit()

    return {"reply": reply}


@app.get("/messages/{session_id}")
async def get_messages(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):

    await find_owned_session(
        session_id,
        current_user,
        db
    )

    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(
            Message.created_at.asc(),
            Message.id.asc()
        )
    )

    messages = result.scalars().all()

    return [
        {
            "role": message.role,
            "content": message.content
        }
        for message in messages
    ]


# ==================================================
# DOCUMENT UPLOAD ENDPOINT
# ==================================================

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required."
        )

    # Extract only the filename, not any supplied directory.
    safe_filename = Path(file.filename).name
    file_extension = Path(safe_filename).suffix.lower()

    if file_extension not in ALLOWED_DOCUMENT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported."
        )

    temp_path = None

    try:
        total_size = 0

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=file_extension,
            delete=False
        ) as temp_file:

            temp_path = temp_file.name

            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds the 10 MB upload limit."
                    )

                temp_file.write(chunk)

        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        logger.info(
            "Document upload started | User: %s | File: %s | Size: %s",
            current_user.email,
            safe_filename,
            total_size
        )

        result = await asyncio.to_thread(
            agent._ingest_document,
            current_user.email,
            temp_path,
            safe_filename
        )

        if (
            not isinstance(result, str)
            or not result.strip()
            or result.startswith("Document ingestion failed.")
        ):
            raise HTTPException(
                status_code=500,
                detail=(
                    "Document ingestion failed. "
                    "Please check the document and try again."
                )
            )

        logger.info(
            "Document upload completed | User: %s | File: %s",
            current_user.email,
            safe_filename
        )

        return {
            "status": "success",
            "filename": safe_filename,
            "size": total_size,
            "message": result
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Document upload failed for user %s",
            current_user.email
        )

        raise HTTPException(
            status_code=500,
            detail="Document upload failed."
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        await file.close()


# --------------------------------------------------
# DIRECT LLM TEST ENDPOINT
# --------------------------------------------------

@app.post("/test-llm")
async def test_llm(
    req: TestLLMRequest,
    current_user: User = Depends(get_current_user)
):

    try:
        response = await llm.invoke(req.prompt)

        return {
            "prompt": req.prompt,
            "response": response,
            "model": llm.model_name
        }

    except Exception as exc:
        logger.exception("Direct LLM test failed")

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# ==================================================
# SERVER-SENT EVENTS HELPERS
# ==================================================

def format_sse_event(event: str, data: dict) -> str:
    """
    Format a Server-Sent Event.
    """

    payload = json.dumps(
        data,
        ensure_ascii=False
    )

    return (
        f"event: {event}\n"
        f"data: {payload}\n\n"
    )


# ==================================================
# STREAMING ENDPOINT
# ==================================================

@app.post("/stream")
async def stream_message(
    req: MessageRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):

    # Verify session ownership.
    session = await find_owned_session(
        req.session_id,
        current_user,
        db
    )

    user_id, agent_session_id = await initialize_agent_session(
        session,
        db
    )

    # Save stable identifiers before entering generator.
    # The generator uses a separate database session.
    database_session_id = session.id

    # Save user message.
    user_msg = Message(
        id=uuid.uuid4(),
        session_id=database_session_id,
        role="user",
        content=req.content
    )

    db.add(user_msg)
    await db.commit()

    logger.info(
        "Starting SSE stream | User: %s | Session: %s",
        user_id,
        agent_session_id
    )

    async def event_generator():
        full_reply = ""
        completed = False

        try:
            async for event in agent.stream_events(
                user_id,
                agent_session_id,
                req.content
            ):
                event_type = event.get("event")
                event_data = event.get("data", {})

                if event_type == "status":
                    logger.info(
                        "Agent status: %s",
                        event_data.get("message", "")
                    )

                    yield format_sse_event(
                        "status",
                        event_data
                    )

                elif event_type == "token":
                    content = event_data.get("content", "")

                    if content:
                        full_reply += content

                        yield format_sse_event(
                            "token",
                            {"content": content}
                        )

                elif event_type == "error":
                    logger.error(
                        "Agent streaming error: %s",
                        event_data.get("message", "")
                    )

                    yield format_sse_event(
                        "error",
                        event_data
                    )

                elif event_type == "done":
                    completed = True
                    logger.info("Agent reported streaming completion.")

            if not completed:
                logger.error(
                    "Agent stream ended without a done event."
                )

                yield format_sse_event(
                    "error",
                    {
                        "message": (
                            "The agent stream ended unexpectedly. "
                            "The response may be incomplete."
                        )
                    }
                )
                return

            if not full_reply.strip():
                full_reply = (
                    "Sorry, the assistant is temporarily unavailable."
                )

            # Use an independent database session.
            async with async_session() as stream_db:
                stream_session = await stream_db.get(
                    Session,
                    database_session_id
                )

                if stream_session is None:
                    logger.error(
                        "Could not find session while saving streamed response."
                    )

                    yield format_sse_event(
                        "error",
                        {"message": "Could not save the conversation."}
                    )
                    return

                # Verify ownership again before persistence.
                if stream_session.user_email != user_id:
                    logger.error(
                        "Session ownership changed during streaming."
                    )

                    yield format_sse_event(
                        "error",
                        {"message": "Session ownership validation failed."}
                    )
                    return

                assistant_msg = Message(
                    id=uuid.uuid4(),
                    session_id=database_session_id,
                    role="assistant",
                    content=full_reply
                )

                stream_db.add(assistant_msg)

                if (
                    user_id in agent.sessions
                    and agent_session_id in agent.sessions[user_id]
                ):
                    stream_session.title = (
                        agent.sessions[user_id][agent_session_id]["title"]
                    )

                await stream_db.commit()

                logger.info(
                    "Assistant response saved successfully | Session: %s",
                    agent_session_id
                )

            yield format_sse_event(
                "done",
                {"message": "Response saved successfully."}
            )

        except Exception as exc:
            logger.exception("SSE streaming failed: %s", exc)

            yield format_sse_event(
                "error",
                {
                    "message": (
                        "An error occurred while streaming the response."
                    )
                }
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
