from pydantic import BaseModel, Field


class SimpleBooleanOutputModelResponse(BaseModel):
    """A simple model for boolean answers."""

    output: bool = Field(..., description="The boolean answer to the question posed by the user.")

class SimpleStringOutputModelResponse(BaseModel):
    """A simple model for string answers."""

    output: str = Field(..., description="The string answer to the question posed by the user.")
