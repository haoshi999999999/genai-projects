import os
import getpass
from urllib.parse import urlparse

import asyncpg
from dotenv import load_dotenv, find_dotenv

env_path = find_dotenv(usecwd=True)
print("Environment file:", env_path or "NOT FOUND")

load_dotenv(env_path)

url = os.environ["DATABASE_URL"].replace(
    "postgresql+asyncpg://",
    "postgresql://"
)

parsed = urlparse(url)
env_password = parsed.password or ""

print("Host:", parsed.hostname)
print("Port:", parsed.port)
print("Database:", parsed.path.lstrip("/"))
print("Username:", parsed.username)
print("Password length loaded from .env:", len(env_password))

manual_password = getpass.getpass("Enter the new PostgreSQL password: ")

print("Environment password character codes:", [ord(c) for c in env_password])
print("Manually entered password character codes:", [ord(c) for c in manual_password])
print("Environment password has whitespace:", env_password != env_password.strip())
print("Passwords match:", env_password == manual_password)

async def test_connection():
    try:
        conn = await asyncpg.connect(
            user=parsed.username,
            password=manual_password,
            host=parsed.hostname,
            port=parsed.port or 5432,
            database=parsed.path.lstrip("/")
        )

        print("Direct Python connection successful!")
        await conn.close()

    except Exception as e:
        print("Direct Python connection failed:", type(e).__name__)
        print(str(e))


import asyncio
asyncio.run(test_connection())