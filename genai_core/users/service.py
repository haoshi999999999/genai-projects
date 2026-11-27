import logging

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from users.interfaces import IUserService
from users.repository import UserRepository
from users.schemas import (
    CreateUserRequest,
    UserResponse,
    UpdateUserRequest,
    DeleteUserResponse,
)
from users.security import hash_password


logger = logging.getLogger(__name__)


class UserService(IUserService):

    def __init__(
        self,
        db: AsyncSession,
        document_manager
    ):
        self.repository = UserRepository(db)
        self.document_manager = document_manager

    async def create_user(
        self,
        request: CreateUserRequest
    ) -> UserResponse:

        email = request.email.strip().lower()

        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email is required."
            )

        if not request.password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password is required."
            )

        existing_user = await self.repository.get_by_email(email)

        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User already exists."
            )

        hashed_password = hash_password(request.password)

        user = await self.repository.create(
            email=email,
            password_hash=hashed_password
        )

        return UserResponse(
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            status="created"
        )

    async def get_user(
        self,
        email: str
    ) -> UserResponse:

        email = email.strip().lower()

        user = await self.repository.get_by_email(email)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found."
            )

        return UserResponse(
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            status="found"
        )

    async def update_user(
        self,
        email: str,
        request: UpdateUserRequest
    ) -> UserResponse:

        email = email.strip().lower()

        user = await self.repository.get_by_email(email)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found."
            )

        # Identify which fields the client actually submitted.
        updates = request.model_dump(exclude_unset=True)

        if not updates:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one field must be provided."
            )

        # Update first name if supplied.
        if "first_name" in updates:

            first_name = updates["first_name"]

            if first_name is None or not first_name.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="First name cannot be empty."
                )

            if len(first_name.strip()) > 100:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="First name cannot exceed 100 characters."
                )

            user.first_name = first_name.strip()

        # Update last name if supplied.
        if "last_name" in updates:

            last_name = updates["last_name"]

            if last_name is None or not last_name.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Last name cannot be empty."
                )

            if len(last_name.strip()) > 100:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Last name cannot exceed 100 characters."
                )

            user.last_name = last_name.strip()

        # Preserve existing password-update functionality.
        if "password" in updates:

            password = updates["password"]

            if password is None or not password:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Password cannot be empty."
                )

            user.password_hash = hash_password(password)

        await self.repository.update(user)

        return UserResponse(
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            status="updated"
        )

    async def delete_user(
        self,
        email: str
    ) -> DeleteUserResponse:

        email = email.strip().lower()

        user = await self.repository.get_by_email(email)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found."
            )

        try:

            # 1. Delete the user's private document store.
            self.document_manager.delete_user_store(email)

            # 2. Delete database records.
            await self.repository.delete(user)

        except Exception:

            logger.exception(
                "Account deletion failed for user: %s",
                email
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Account deletion failed."
            )

        return DeleteUserResponse(
            email=email,
            status="deleted"
        )