import os
import uuid
import hashlib
import traceback
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Query
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
        logger.info("Groq response content: %r", content)

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
    allow_origins=["*"],
    allow_credentials=True,
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
    return hashlib.sha256(password.encode()).hexdigest()


async def find_session(session_id: uuid.UUID, db: AsyncSession):
    result = await db.execute(select(Session).where(Session.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


async def initialize_agent_session(session: Session, db: AsyncSession):
    user_id = session.user_email
    session_id = str(session.id)

    if user_id not in agent.sessions:
        agent.sessions[user_id] = {}

    if session_id in agent.sessions[user_id]:
        agent.sessions[user_id][session_id]["persona"] = session.persona
        return user_id, session_id

    result = await db.execute(
        select(Message).where(Message.session_id == session.id).order_by(Message.id)
    )
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
    email: str
    persona: str = "default"
    title: str = "New Chat"


class MessageRequest(BaseModel):
    session_id: uuid.UUID
    content: str


class PersonaUpdate(BaseModel):
    session_id: uuid.UUID
    persona: str


# --------------------------------------------------
# Authentication endpoints
# --------------------------------------------------

@app.post("/register")
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == req.email))
    existing_user = result.scalar_one_or_none()
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")
    user = User(email=req.email, password_hash=hash_password(req.password))
    db.add(user)
    await db.commit()
    return {"status": "registered"}


@app.post("/login")
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == req.email))
    user = result.scalar_one_or_none()
    if not user or user.password_hash != hash_password(req.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"status": "logged_in"}


# --------------------------------------------------
# Session endpoints
# --------------------------------------------------

# Static route must be registered before dynamic route to avoid routing conflicts.
@app.get("/session/list")
async def list_sessions(email: str = Query(...), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Session).where(Session.user_email == email))
    sessions = result.scalars().all()
    return {
        str(s.id): {"title": s.title, "persona": s.persona}
        for s in sessions
    }


@app.post("/session/create")
async def create_session(req: CreateSessionRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == req.email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if req.persona not in agent.personas:
        raise HTTPException(status_code=400, detail="Invalid persona")
    new_session = Session(id=uuid.uuid4(), user_email=req.email, persona=req.persona, title=req.title)
    db.add(new_session)
    await db.commit()
    await db.refresh(new_session)
    return {"session_id": new_session.id}


@app.get("/session/{session_id}")
async def get_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    session = await find_session(session_id, db)
    return {"id": session.id, "title": session.title, "persona": session.persona, "user_email": session.user_email}


@app.post("/session/persona")
async def update_persona(req: PersonaUpdate, db: AsyncSession = Depends(get_db)):
    session = await find_session(req.session_id, db)
    if req.persona not in agent.personas:
        raise HTTPException(status_code=400, detail="Invalid persona")
    session.persona = req.persona
    await db.commit()
    user_id = session.user_email
    session_id = str(session.id)
    if user_id in agent.sessions and session_id in agent.sessions[user_id]:
        agent.sessions[user_id][session_id]["persona"] = req.persona
    return {"status": "updated"}


@app.post("/session/clear")
async def clear_session(req: MessageRequest, db: AsyncSession = Depends(get_db)):
    session = await find_session(req.session_id, db)
    #await db.execute(Message.__table__.delete().where(Message.session_id == req.req.session_id if False else Message.session_id == req.session_id))
    await db.execute(
    Message.__table__.delete().where(
        Message.session_id == req.session_id
    )
)
    
    await db.commit()
    user_id = session.user_email
    session_id = str(session.id)
    if user_id in agent.sessions and session_id in agent.sessions[user_id]:
        agent.sessions[user_id][session_id]["messages"] = []
        agent.sessions[user_id][session_id]["title"] = "New Chat"
    return {"status": "cleared"}


# --------------------------------------------------
# Chat endpoints
# --------------------------------------------------

@app.post("/message/send")
async def send_message(req: MessageRequest, db: AsyncSession = Depends(get_db)):
    session = await find_session(req.session_id, db)
    user_id, agent_session_id = await initialize_agent_session(session, db)

    user_msg = Message(id=uuid.uuid4(), session_id=session.id, role="user", content=req.content)
    db.add(user_msg)
    await db.commit()

    try:
        reply = await agent.run(user_id, agent_session_id, req.content)
        if not isinstance(reply, str) or not reply.strip():
           raise RuntimeError("Agent returned an empty response.")
    except Exception as exc:
        logger.error("Agent.run failed: %s", exc)
        logger.error(traceback.format_exc())
        # Fallback reply so UI doesn't get a 500 while debugging
        reply = "Sorry, the assistant is temporarily unavailable."

    assistant_msg = Message(id=uuid.uuid4(), session_id=session.id, role="assistant", content=reply)
    db.add(assistant_msg)
    await db.commit()

    try:
        session.title = agent.sessions[user_id][agent_session_id]["title"]
        await db.commit()
    except Exception:
        logger.debug("Could not persist session title; agent state missing.")

    return {"reply": reply}


@app.get("/messages/{session_id}")
async def get_messages(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    await find_session(session_id, db)
    result = await db.execute(select(Message).where(Message.session_id == session_id).order_by(Message.id))
    messages = result.scalars().all()
    return [{"role": m.role, "content": m.content} for m in messages]

# Temporary endpoint to test Groq independently of Agent
@app.post("/test-llm")
async def test_llm(req: TestLLMRequest):
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
async def stream_message(req: MessageRequest, db: AsyncSession = Depends(get_db)):
    session = await find_session(req.session_id, db)
    user_id, agent_session_id = await initialize_agent_session(session, db)

    user_msg = Message(id=uuid.uuid4(), session_id=session.id, role="user", content=req.content)
    db.add(user_msg)
    await db.commit()

    async def token_generator():
        full_reply = ""
        try:
            async for token in agent.stream(user_id, agent_session_id, req.content):
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
                logger.error("Failed to save fallback assistant message: %s", traceback.format_exc())
            yield fallback
            return

        assistant_msg = Message(id=uuid.uuid4(), session_id=session.id, role="assistant", content=full_reply)
        db.add(assistant_msg)
        await db.commit()

        try:
            session.title = agent.sessions[user_id][agent_session_id]["title"]
            await db.commit()
        except Exception:
            logger.debug("Could not persist session title after streaming; agent state missing.")

    return StreamingResponse(token_generator(), media_type="text/plain")
