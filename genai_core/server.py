import os
import uuid
import hashlib
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
    File,
    Form
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


# ==================================================
# 1. LOGGING
# ==================================================

logging.basicConfig(
    level=logging.INFO
)

logger = logging.getLogger(__name__)


# ==================================================
# 2. CONFIGURATION
# ==================================================

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


# ==================================================
# 3. AUTHENTICATION CONFIGURATION
# ==================================================

SECRET_KEY = os.getenv("JWT_SECRET_KEY")

if not SECRET_KEY:

    raise RuntimeError(
        "JWT_SECRET_KEY is missing from .env"
    )

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 60

password_hash = PasswordHash.recommended()

security = HTTPBearer(
    auto_error=False
)


# ==================================================
# 4. DOCUMENT UPLOAD CONFIGURATION
# ==================================================

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

ALLOWED_DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".txt"
}

ALLOWED_DOCUMENT_AUTHORITIES = {
    "Official Policy",
    "Employee Handbook",
    "Reference Material",
    "Informational"
}


# ==================================================
# 5. GROQ CLIENT WRAPPER
# ==================================================

class GroqLLM:

    def __init__(
        self,
        api_key: str,
        model_name: str
    ):

        self.client = AsyncGroq(
            api_key=api_key
        )

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

            logger.error(
                "Model not found: %s",
                self.model_name
            )

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

        logger.info(
            "Groq model: %s",
            self.model_name
        )

        if not content or not content.strip():

            raise RuntimeError(
                "Groq returned an empty response."
            )

        return content.strip()


# ==================================================
# 6. INITIALIZE LLM AND AGENT
# ==================================================

llm = GroqLLM(
    api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL
)

agent = Agent(
    llm=llm
)


# ==================================================
# 7. FASTAPI APPLICATION
# ==================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(
        "Starting GenAI Agent API."
    )

    yield

    logger.info(
        "Shutting down GenAI Agent API."
    )


app = FastAPI(
    lifespan=lifespan
)

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


# ==================================================
# 8. DATABASE HELPERS
# ==================================================

async def get_db():

    async with async_session() as db:

        yield db


# ==================================================
# 9. PASSWORD HELPERS
# ==================================================

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
            password.encode("utf-8")
        ).hexdigest()

        return secrets.compare_digest(
            legacy_hash,
            hashed_password
        )

    return False


# ==================================================
# 10. JWT HELPERS
# ==================================================

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


# ==================================================
# 11. AUTHENTICATION DEPENDENCY
# ==================================================

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
        select(User).where(
            User.email == email
        )
    )

    user = result.scalar_one_or_none()

    if not user:

        raise HTTPException(
            status_code=401,
            detail="User not found"
        )

    return user


# ==================================================
# 12. SESSION OWNERSHIP VALIDATION
# ==================================================

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


# ==================================================
# 13. INITIALIZE AGENT SESSION
# ==================================================

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
        .where(
            Message.session_id == session.id
        )
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


# ==================================================
# 14. REQUEST MODELS
# ==================================================

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


# ==================================================
# 15. AUTHENTICATION ENDPOINTS
# ==================================================

@app.post("/register")
async def register(
    req: RegisterRequest,
    db: AsyncSession = Depends(get_db)
):

    email = req.email.strip().lower()

    if not email:

        raise HTTPException(
            status_code=400,
            detail="Email is required."
        )

    if not req.password:

        raise HTTPException(
            status_code=400,
            detail="Password is required."
        )

    result = await db.execute(
        select(User).where(
            User.email == email
        )
    )

    existing_user = result.scalar_one_or_none()

    if existing_user:

        raise HTTPException(
            status_code=400,
            detail="User already exists"
        )

    user = User(
        email=email,
        password_hash=hash_password(req.password)
    )

    db.add(user)

    await db.commit()

    return {
        "status": "registered"
    }


@app.post("/login")
async def login(
    req: LoginRequest,
    db: AsyncSession = Depends(get_db)
):

    email = req.email.strip().lower()

    result = await db.execute(
        select(User).where(
            User.email == email
        )
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

        user.password_hash = hash_password(
            req.password
        )

        await db.commit()

    access_token = create_access_token(
        user.email
    )

    return {

        "status": "logged_in",

        "access_token": access_token,

        "token_type": "bearer"

    }


# ==================================================
# 16. SESSION ENDPOINTS
# ==================================================

@app.get("/session/list")
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):

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
        .group_by(
            Session.id
        )
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

    return {
        "status": "updated"
    }


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

        agent.sessions[user_id][agent_session_id]["title"] = (
            "New Chat"
        )

    return {

        "status": "cleared",

        "session_id": str(session.id),

        "title": "New Chat"

    }


# ==================================================
# 17. CHAT ENDPOINTS
# ==================================================

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

            raise RuntimeError(
                "Agent returned an empty response."
            )

    except Exception as exc:

        logger.exception(
            "Agent.run failed: %s",
            exc
        )

        reply = (
            "Sorry, the assistant is temporarily unavailable."
        )

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

    return {
        "reply": reply
    }


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
        .where(
            Message.session_id == session_id
        )
        .order_by(
            Message.created_at.asc(),
            Message.id.asc()
        )
    )

    messages = result.scalars().all()

    logger.info(
        "MESSAGE HISTORY | User: %s | Session: %s | Total messages: %s",
        current_user.email,
        session_id,
        len(messages)
    )

    return [

        {
            "role": message.role,
            "content": message.content
        }

        for message in messages

    ]


# ==================================================
# 18. DOCUMENT UPLOAD ENDPOINT
# ==================================================

@app.post("/documents/upload")
async def upload_document(

    # Uploaded document
    file: UploadFile = File(...),

    # Document metadata
    document_name: str | None = Form(default=None),

    document_version: str | None = Form(default=None),

    effective_date: str | None = Form(default=None),

    authority: str | None = Form(default=None),

    owner_department: str | None = Form(default=None),

    # Authenticated user
    current_user: User = Depends(get_current_user)

):

    # --------------------------------------------------
    # 1. Validate filename
    # --------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="A filename is required."
        )

    # Handle both Windows and Unix path separators.
    safe_filename = (
        file.filename.replace("\\", "/").split("/")[-1]
    )

    if not safe_filename or safe_filename in {".", ".."}:

        raise HTTPException(
            status_code=400,
            detail="Invalid filename."
        )

    file_extension = Path(
        safe_filename
    ).suffix.lower()

    if file_extension not in ALLOWED_DOCUMENT_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported."
        )

    # --------------------------------------------------
    # 2. Validate document metadata
    # --------------------------------------------------

    document_name = (
        document_name.strip()
        if document_name and document_name.strip()
        else None
    )

    document_version = (
        document_version.strip()
        if document_version and document_version.strip()
        else None
    )

    effective_date = (
        effective_date.strip()
        if effective_date and effective_date.strip()
        else None
    )

    authority = (
        authority.strip()
        if authority and authority.strip()
        else None
    )

    owner_department = (
        owner_department.strip()
        if owner_department and owner_department.strip()
        else None
    )

    # Validate authority.
    if (
        authority is not None
        and authority not in ALLOWED_DOCUMENT_AUTHORITIES
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid authority. Allowed values: "
                f"{sorted(ALLOWED_DOCUMENT_AUTHORITIES)}"
            )
        )

    # Validate effective date.
    if effective_date is not None:

        try:

            parsed_date = datetime.strptime(
                effective_date,
                "%Y-%m-%d"
            )

            if parsed_date.strftime("%Y-%m-%d") != effective_date:
                raise ValueError

        except ValueError:

            raise HTTPException(
                status_code=400,
                detail="effective_date must use YYYY-MM-DD format."
            )

    # --------------------------------------------------
    # 3. Prepare temporary file
    # --------------------------------------------------

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

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_UPLOAD_SIZE:

                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds the 10 MB upload limit."
                    )

                temp_file.write(chunk)

        # Reject empty files.
        if total_size == 0:

            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        # --------------------------------------------------
        # 4. Log upload information
        # --------------------------------------------------

        logger.info(
            "Document upload started | "
            "User: %s | File: %s | Size: %s | "
            "Name: %s | Version: %s | Effective Date: %s | "
            "Authority: %s | Department: %s",
            current_user.email,
            safe_filename,
            total_size,
            document_name,
            document_version,
            effective_date,
            authority,
            owner_department
        )

        # --------------------------------------------------
        # 5. Ingest document with metadata
        # --------------------------------------------------

        result = await asyncio.to_thread(

            agent._ingest_document,

            current_user.email,

            temp_path,

            safe_filename,

            document_name=document_name,

            document_version=document_version,

            effective_date=effective_date,

            authority=authority,

            owner_department=owner_department

        )

        # --------------------------------------------------
        # 6. Validate ingestion result
        # --------------------------------------------------

        if not isinstance(result, str) or not result.strip():

            logger.error(
                "Document ingestion returned an invalid result | "
                "User: %s | File: %s | Result: %s",
                current_user.email,
                safe_filename,
                result
            )

            raise HTTPException(
                status_code=422,
                detail=(
                    "Document ingestion failed. "
                    "Please verify that the document contains readable text."
                )
            )

        successful_messages = (
            "Document uploaded and indexed successfully:",
            "Document already exists with identical content",
            "Document metadata updated successfully."
        )

        if not result.startswith(successful_messages):

            logger.error(
                "Document ingestion unsuccessful | "
                "User: %s | File: %s | Result: %s",
                current_user.email,
                safe_filename,
                result
            )

            raise HTTPException(
                status_code=422,
                detail=result
            )

        # --------------------------------------------------
        # 7. Log successful upload
        # --------------------------------------------------

        logger.info(
            "Document upload completed | "
            "User: %s | File: %s",
            current_user.email,
            safe_filename
        )

        # --------------------------------------------------
        # 8. Return upload result
        # --------------------------------------------------

        return {

            "status": "success",

            "filename": safe_filename,

            "size": total_size,

            "metadata": {

                "document_name": document_name or safe_filename,

                "document_version": document_version,

                "effective_date": effective_date,

                "authority": authority,

                "upload_date": datetime.now(
                    timezone.utc
                ).isoformat(),

                "owner_department": owner_department

            },

            "message": result

        }

    # --------------------------------------------------
    # 9. Error handling
    # --------------------------------------------------

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

    # --------------------------------------------------
    # 10. Cleanup
    # --------------------------------------------------

    finally:

        if temp_path and os.path.exists(temp_path):

            try:

                os.remove(temp_path)

            except OSError:

                logger.exception(
                    "Failed to remove temporary upload file."
                )

        await file.close()


# ==================================================
# 19. DOCUMENT LIST ENDPOINT
# ==================================================

@app.get("/documents/list")
async def list_documents(
    current_user: User = Depends(get_current_user)
):

    try:

        documents = await asyncio.to_thread(

            agent.docs.list_documents,

            user_id=current_user.email

        )

        return {

            "status": "success",

            "count": len(documents),

            "documents": documents

        }

    except Exception:

        logger.exception(
            "Failed to list documents for user %s",
            current_user.email
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve documents."
        )


# ==================================================
# 20. DIRECT LLM TEST ENDPOINT
# ==================================================

@app.post("/test-llm")
async def test_llm(
    req: TestLLMRequest,
    current_user: User = Depends(get_current_user)
):

    try:

        response = await llm.invoke(
            req.prompt
        )

        return {

            "prompt": req.prompt,

            "response": response,

            "model": llm.model_name

        }

    except Exception:

        logger.exception(
            "Direct LLM test failed"
        )

        raise HTTPException(
            status_code=500,
            detail="LLM request failed. Please try again."
        )


# ==================================================
# 21. SERVER-SENT EVENTS HELPERS
# ==================================================

def format_sse_event(
    event: str,
    data: dict
) -> str:

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
# 22. STREAMING ENDPOINT
# ==================================================

@app.post("/stream")
async def stream_message(
    req: MessageRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):

    # --------------------------------------------------
    # 1. Verify session ownership
    # --------------------------------------------------

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
    database_session_id = session.id

    # --------------------------------------------------
    # 2. Save user message
    # --------------------------------------------------

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

    # --------------------------------------------------
    # 3. SSE event generator
    # --------------------------------------------------

    async def event_generator():

        full_reply = ""

        completed = False

        failed = False

        try:

            async for event in agent.stream_events(
                user_id,
                agent_session_id,
                req.content
            ):

                event_type = event.get("event")

                event_data = event.get("data", {})

                # ------------------------------------------
                # Status event
                # ------------------------------------------

                if event_type == "status":

                    logger.info(
                        "Agent status: %s",
                        event_data.get("message", "")
                    )

                    yield format_sse_event(
                        "status",
                        event_data
                    )

                # ------------------------------------------
                # Token event
                # ------------------------------------------

                elif event_type == "token":

                    content = event_data.get(
                        "content",
                        ""
                    )

                    if content:

                        full_reply += content

                        yield format_sse_event(
                            "token",
                            {
                                "content": content
                            }
                        )

                # ------------------------------------------
                # Error event
                # ------------------------------------------

                elif event_type == "error":

                    failed = True

                    logger.error(
                        "Agent streaming error: %s",
                        event_data.get("message", "")
                    )

                    yield format_sse_event(
                        "error",
                        event_data
                    )

                # ------------------------------------------
                # Completion event
                # ------------------------------------------

                elif event_type == "done":

                    completed = True

                    logger.info(
                        "Agent reported streaming completion."
                    )

            # --------------------------------------------------
            # 4. Validate streaming result
            # --------------------------------------------------

            if failed:

                logger.error(
                    "Agent streaming failed. "
                    "Assistant response will not be saved."
                )

                return

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

            # --------------------------------------------------
            # 5. Persist assistant response
            # --------------------------------------------------

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
                        {
                            "message": "Could not save the conversation."
                        }
                    )

                    return

                # Verify ownership again.
                if stream_session.user_email != user_id:

                    logger.error(
                        "Session ownership changed during streaming."
                    )

                    yield format_sse_event(
                        "error",
                        {
                            "message": (
                                "Session ownership validation failed."
                            )
                        }
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

            # --------------------------------------------------
            # 6. Send final completion event
            # --------------------------------------------------

            yield format_sse_event(
                "done",
                {
                    "message": "Response saved successfully."
                }
            )

        except Exception:

            logger.exception(
                "SSE streaming failed."
            )

            yield format_sse_event(
                "error",
                {
                    "message": (
                        "An error occurred while streaming the response."
                    )
                }
            )

    # --------------------------------------------------
    # 7. Return streaming response
    # --------------------------------------------------

    return StreamingResponse(

        event_generator(),

        media_type="text/event-stream",

        headers={

            "Cache-Control": "no-cache",

            "Connection": "keep-alive",

            "X-Accel-Buffering": "no"

        }

    )