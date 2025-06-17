from enum import Enum
from typing import List, Optional

import polars as pl
from pydantic import BaseModel, Field

from fenic._backends.local.semantic_operators.extract import Extract


class StatusEnum(str, Enum):
    """Enumeration for the status of a task."""

    TODO = "To Do"
    IN_PROGRESS = "In Progress"
    DONE = "Done"


class SubTask(BaseModel):
    """Model representing a sub-task of a main task."""

    title: str = Field(..., description="Title of the sub-task.")
    completed: bool = Field(..., description="Whether the sub-task has been completed.")


class Task(BaseModel):
    """Model representing a task in a to-do application."""

    id: int = Field(..., description="Unique identifier for the task.")
    title: str = Field(..., description="Title or name of the task.")
    description: Optional[str] = Field(
        None, description="Detailed description of the task."
    )
    tags: List[str] = Field(..., description="List of tags associated with the task.")
    status: StatusEnum = Field(..., description="Current status of the task.")
    subtasks: List[SubTask] = Field(
        ..., description="List of sub-tasks related to this task."
    )


class TaskList(BaseModel):
    """Container model holding a list of tasks."""

    owner: str = Field(..., description="Username of the task list owner.")
    tasks: List[Task] = Field(..., description="List of tasks belonging to the user.")


class TestExtract:
    """Test cases for the Extract operator."""

    def test_build_prompts(self, local_session):
        document = """
Alice has a task list containing one main task. The task has the ID `1` and is titled **"Build Pydantic Model"**. Its purpose is to **create a sample with nested structures, enums, and lists**.

This task is currently marked as **"In Progress"**. It is associated with the tags: **python**, **pydantic**, and **modeling**.

There are two subtasks under this main task:
1. **"Write model"** – this subtask has already been completed.
2. **"Add descriptions"** – this subtask is still pending.

        The task list belongs to the user **alice**.
        """

        input = pl.Series("input", [document])

        extract = Extract(
            input=input,
            schema=TaskList,
            model=local_session._session_state.get_language_model(),
            max_output_tokens=1024,
            temperature=0,
        )

        result = list(
            map(
                lambda x: x.to_message_list() if x else None,
                extract.build_request_messages_batch(),
            )
        )
        expected = [
            [
                {
                    "content": "You are an expert at structured data extraction. Your task is "
                    "to extract relevant information from a given document. Your "
                    "output must be a structured JSON object. Expected JSON keys and "
                    "descriptions:\n"
                    "owner (str): Username of the task list owner.\n"
                    "tasks (list of objects): List of tasks belonging to the user.\n"
                    "tasks[item].id (int): Unique identifier for the task.\n"
                    "tasks[item].title (str): Title or name of the task.\n"
                    "tasks[item].description (str (optional)): Detailed description "
                    "of the task.\n"
                    "tasks[item].tags (list of str): List of tags associated with "
                    "the task.\n"
                    "tasks[item].status (StatusEnum): Current status of the task.\n"
                    "tasks[item].subtasks (list of objects): List of sub-tasks "
                    "related to this task.\n"
                    "tasks[item].subtasks[item].title (str): Title of the sub-task.\n"
                    "tasks[item].subtasks[item].completed (bool): Whether the "
                    "sub-task has been completed.Notes on the structure:\n"
                    "- Field names with parent.child notation indicate nested "
                    "objects\n"
                    "- [item] notation indicates items within a list\n"
                    "- Type information is provided in parentheses\n",
                    "role": "system",
                },
                {
                    "content": "\n"
                    "Alice has a task list containing one main task. The task has "
                    'the ID `1` and is titled **"Build Pydantic Model"**. Its '
                    "purpose is to **create a sample with nested structures, enums, "
                    "and lists**.\n"
                    "\n"
                    'This task is currently marked as **"In Progress"**. It is '
                    "associated with the tags: **python**, **pydantic**, and "
                    "**modeling**.\n"
                    "\n"
                    "There are two subtasks under this main task:\n"
                    '1. **"Write model"** – this subtask has already been '
                    "completed.\n"
                    '2. **"Add descriptions"** – this subtask is still pending.\n'
                    "\n"
                    "        The task list belongs to the user **alice**.\n"
                    "        ",
                    "role": "user",
                },
            ]
        ]
        assert result == expected
