from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import User
from users.schemas import (
    UserResponse,
    UpdateUserRequest,
    DeleteUserResponse,
)
from users.service import UserService


def create_users_router(
    get_current_user_dependency,
    get_db_dependency,
    document_manager
):

    router = APIRouter(
        prefix="/users",
        tags=["Users"]
    )

    @router.get(
        "/me",
        response_model=UserResponse
    )
    async def get_my_account(
        current_user: User = Depends(
            get_current_user_dependency
        ),
        db: AsyncSession = Depends(
            get_db_dependency
        )
    ):

        service = UserService(
            db,
            document_manager
        )

        return await service.get_user(
            current_user.email
        )

    @router.patch(
        "/me",
        response_model=UserResponse
    )
    async def update_my_account(
        request: UpdateUserRequest,
        current_user: User = Depends(
            get_current_user_dependency
        ),
        db: AsyncSession = Depends(
            get_db_dependency
        )
    ):

        service = UserService(
            db,
            document_manager
        )

        return await service.update_user(
            current_user.email,
            request
        )

    @router.delete(
        "/me",
        response_model=DeleteUserResponse
    )
    async def delete_my_account(
        current_user: User = Depends(
            get_current_user_dependency
        ),
        db: AsyncSession = Depends(
            get_db_dependency
        )
    ):

        service = UserService(
            db,
            document_manager
        )

        return await service.delete_user(
            current_user.email
        )

    return router
