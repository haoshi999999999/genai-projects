import os
import uuid
import hashlib
import traceback
import bcrypt
from contextlib import asynccontextmanager

import jwt
import secrets
from datetime import datetime, timedelta, timezone

from pwdlib import PasswordHash
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from groq import AsyncGroq, NotFoundError

from db.database import async_session
from db.models import User, Session, Message
from agent.agent import Agent

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
FALLBACK_GROQ_MODEL = os.getenv("FALLBACK_GROQ_MODEL", None)

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file or environment.")


# --------------------------------------------------
# Authentication configuration
# --------------------------------------------------

SECRET_KEY = os.getenv("JWT_SECRET_KEY")

if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY is missing from .env")

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 60

password_hash = PasswordHash.recommended()

security = HTTPBearer(auto_error=False)

# --------------------------------------------------
# Groq client wrapper
# --------------------------------------------------

class GroqLLM:
    def __init__(self, api_key: str, model_name: str):
        self.client = AsyncGroq(api_key=api_key)
        self.chat = self.client.chat
        self.model_name = model_name

    async def invoke(self, prompt: str, *, stream: bool = False):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer clearly and concisely."
            },
            {"role": "user", "content": prompt}
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

            logger.info("Retrying with fallback model: %s", FALLBACK_GROQ_MODEL)

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
    

llm = GroqLLM(api_key=GROQ_API_KEY, model_name=GROQ_MODEL)
agent = Agent(llm=llm)


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Database helpers
# --------------------------------------------------

async def get_db():
    async with async_session() as db:
        yield db


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:

    # Argon2 password hashes
    if hashed_password.startswith("$argon2"):
        return password_hash.verify(password, hashed_password)

    # Bcrypt password hashes
    if hashed_password.startswith(("$2a$", "$2b$", "$2y$")):
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


def create_access_token(email: str):

    expire = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": email,
        "exp": expire
    }

    token = jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM
    )

    return token


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


async def initialize_agent_session(session: Session, db: AsyncSession):
    user_id = session.user_email
    session_id = str(session.id)

    if user_id not in agent.sessions:
        agent.sessions[user_id] = {}

    if session_id in agent.sessions[user_id]:
        agent.sessions[user_id][session_id]["persona"] = session.persona
        return user_id, session_id

    result = await db.execute(select(Message).where(Message.session_id == session.id).order_by(Message.created_at.asc(), Message.id.asc()))
    stored_messages = result.scalars().all()

    agent.sessions[user_id][session_id] = {
        "title": session.title,
        "persona": session.persona,
        "messages": [{"role": m.role, "content": m.content} for m in stored_messages],
    }

    return user_id, session_id


# --------------------------------------------------
# Request models
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
# Authentication endpoints
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

    # Upgrade legacy SHA-256 hashes to Argon2
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
# Session endpoints
# --------------------------------------------------
@app.get("/session/list")
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):

    result = await db.execute(
        select(Session).where(
            Session.user_email == current_user.email
        )
    )

    sessions = result.scalars().all()

    return {
        str(s.id): {
            "title": s.title,
            "persona": s.persona
        }
        for s in sessions
    }


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

    return {"session_id": new_session.id}


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

    if user_id in agent.sessions and session_id in agent.sessions[user_id]:
        agent.sessions[user_id][session_id]["persona"] = req.persona

    return {"status": "updated"}


@app.post("/session/clear")
async def clear_session(
    req: SessionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # 1. Verify that the session belongs to the authenticated user
    session = await find_owned_session(
        req.session_id,
        current_user,
        db
    )

    # 2. Delete all messages associated with this session
    await db.execute(
        Message.__table__.delete().where(
            Message.session_id == session.id
        )
    )

    # 3. Reset the session title
    session.title = "New Chat"

    # 4. Persist changes to PostgreSQL
    await db.commit()

    # 5. Reset the corresponding in-memory Agent conversation
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
# Chat endpoints
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

    assistant_msg = Message(
        id=uuid.uuid4(),
        session_id=session.id,
        role="assistant",
        content=reply
    )

    db.add(assistant_msg)

    await db.commit()

    session.title = agent.sessions[user_id][agent_session_id]["title"]

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
        .order_by(Message.created_at.asc(), Message.id.asc())
    )

    messages = result.scalars().all()

    return [
        {
            "role": m.role,
            "content": m.content
        }
        for m in messages
    ]

# Temporary endpoint to test Groq independently of Agent
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
        raise HTTPException(status_code=500, detail=str(exc))

# --------------------------------------------------
# Streaming endpoint
# --------------------------------------------------

@app.post("/stream")
async def stream_message(
    req: MessageRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # Verify that this session belongs to the authenticated user
    session = await find_owned_session(
        req.session_id,
        current_user,
        db
    )

    user_id, agent_session_id = await initialize_agent_session(
        session,
        db
    )

    user_msg = Message(
        id=uuid.uuid4(),
        session_id=session.id,
        role="user",
        content=req.content
    )

    db.add(user_msg)
    await db.commit()

    async def token_generator():
        full_reply = ""

        try:
            async for token in agent.stream(
                user_id,
                agent_session_id,
                req.content
            ):
                full_reply += token
                yield token

        except Exception as exc:
            logger.error("Agent.stream failed: %s", exc)
            logger.error(traceback.format_exc())

            fallback = "\n[Error generating response.]"

            try:
                assistant_msg = Message(
                    id=uuid.uuid4(),
                    session_id=session.id,
                    role="assistant",
                    content="Sorry, the assistant is temporarily unavailable."
                )

                db.add(assistant_msg)
                await db.commit()

            except Exception:
                logger.error(
                    "Failed to save fallback assistant message: %s",
                    traceback.format_exc()
                )

            yield fallback
            return

        assistant_msg = Message(
            id=uuid.uuid4(),
            session_id=session.id,
            role="assistant",
            content=full_reply
        )

        db.add(assistant_msg)
        await db.commit()

        try:
            session.title = agent.sessions[user_id][agent_session_id]["title"]
            await db.commit()

        except Exception:
            logger.debug(
                "Could not persist session title after streaming; agent state missing."
            )

    return StreamingResponse(
        token_generator(),
        media_type="text/plain"
    )