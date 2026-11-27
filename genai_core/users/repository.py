from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import User, Session, Message


class UserRepository:

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_email(self, email: str):
        result = await self.db.execute(
            select(User).where(User.email == email)
        )

        return result.scalar_one_or_none()

    async def create(
        self,
        email: str,
        password_hash: str
    ):
        user = User(
            email=email,
            password_hash=password_hash
        )

        self.db.add(user)

        await self.db.commit()
        await self.db.refresh(user)

        return user

    async def update(self, user: User):
        await self.db.commit()
        await self.db.refresh(user)

        return user

    async def delete(self, user: User):

        try:

            # 1. Delete messages belonging to this user's sessions.
            await self.db.execute(
                delete(Message).where(
                    Message.session_id.in_(
                        select(Session.id).where(
                            Session.user_email == user.email
                        )
                    )
                )
            )

            # 2. Delete the user's sessions.
            await self.db.execute(
                delete(Session).where(
                    Session.user_email == user.email
                )
            )

            # 3. Delete the user account.
            await self.db.delete(user)

            # Commit all database deletions together.
            await self.db.commit()

            return True

        except Exception:

            await self.db.rollback()

            raise