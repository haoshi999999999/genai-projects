import os

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

# IMPORTANT:
# Change username/password/host/dbname to match your PostgreSQL setup.
# DATABASE_URL = "postgresql+asyncpg://postgres:0921@localhost:5432/genai_agent"



# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=True,            # shows SQL in terminal (helpful for debugging)
    future=True
)

# Create async session factory
async_session = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# FastAPI dependency
async def get_db():
    async with async_session() as session:
        yield session
