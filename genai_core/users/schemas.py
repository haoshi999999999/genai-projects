from pydantic import BaseModel


class CreateUserRequest(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    email: str
    first_name: str | None = None
    last_name: str | None = None
    status: str


class UpdateUserRequest(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    password: str | None = None


class DeleteUserResponse(BaseModel):
    email: str
    status: str