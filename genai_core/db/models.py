from sqlalchemy import Column, String, Text, ForeignKey
from sqlalchemy import DateTime, func
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    email = Column(String, primary_key=True)
    password_hash = Column(String, nullable=False)

    sessions = relationship("Session", back_populates="user")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_email = Column(String, ForeignKey("users.email"))
    title = Column(String, default="New Chat")
    persona = Column(String, default="default")

    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session")


class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"))
    role = Column(String)     # "user" or "assistant"
    content = Column(Text)
    created_at = Column(
                        DateTime(timezone=True),
                        server_default=func.now(),
                        nullable=False,
                    )

    session = relationship("Session", back_populates="messages")
