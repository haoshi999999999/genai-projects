from abc import ABC, abstractmethod

from users.schemas import (
    CreateUserRequest,
    UserResponse,
    UpdateUserRequest,
    DeleteUserResponse,
)


class IUserService(ABC):

    @abstractmethod
    async def create_user(
        self,
        request: CreateUserRequest
    ) -> UserResponse:
        pass

    @abstractmethod
    async def get_user(
        self,
        email: str
    ) -> UserResponse:
        pass

    @abstractmethod
    async def update_user(
        self,
        email: str,
        request: UpdateUserRequest
    ) -> UserResponse:
        pass

    @abstractmethod
    async def delete_user(
        self,
        email: str
    ) -> DeleteUserResponse:
        pass