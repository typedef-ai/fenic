"""Global progress tracking for model client operations using Rich."""

from __future__ import annotations

import threading
from typing import Optional
from fenic.logging import _shared_console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
    TaskID,
)


class ProgressManager:
    """Singleton manager for Rich progress displays across multiple threads."""

    _instance: Optional[ProgressManager] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        """Initialize the Rich progress display."""
        self.console = _shared_console
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TextColumn("• {task.fields[extra]}"),
            console=self.console,
            transient=False,
            refresh_per_second=10,
        )
        self._context_count = 0
        self._context_lock = threading.Lock()

    def __enter__(self):
        """Start the progress display if not already started."""
        with self._context_lock:
            if self._context_count == 0 and not self.progress.live.is_started:
                self.progress.start()
            self._context_count += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the progress display when all contexts are done."""
        with self._context_lock:
            self._context_count = max(0, self._context_count - 1)
            if self._context_count == 0 and self.progress.live.is_started:
                self.progress.stop()

    def add_task(self, description: str, total: int, **fields) -> TaskID:
        """Add a new task to the progress display.

        Args:
            description: Human-readable task description.
            total: Total units of work for the task. If <= 0, will default to 1.
            extra (str, optional): Custom annotation shown on the right.

        Returns:
            TaskID: An identifier for the created task.
        """
        if total <= 0:
            total = 1
        if 'extra' not in fields:
            fields['extra'] = ""
        return self.progress.add_task(description, total=total, **fields)

    def update(self, task_id: TaskID, advance: int = 1, **fields):
        """Update a task's progress."""
        self.progress.update(task_id, advance=advance, **fields)

    def remove_task(self, task_id: TaskID):
        """Remove a task from the progress display."""
        self.progress.remove_task(task_id)


# Global instance
_progress_manager = ProgressManager()


def get_progress_manager() -> ProgressManager:
    """Get the global progress manager instance."""
    return _progress_manager
