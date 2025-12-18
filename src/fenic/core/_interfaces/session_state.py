from abc import ABC, abstractmethod

from fenic.core._interfaces.catalog import BaseCatalog
from fenic.core._interfaces.execution import BaseExecution
from fenic.core._resolved_session_config import ResolvedSessionConfig


class BaseSessionState(ABC):
    def __init__(self, session_config: ResolvedSessionConfig):
        self.session_config = session_config

    @property
    @abstractmethod
    def execution(self) -> BaseExecution:
        """Access the execution interface."""
        pass

    @property
    @abstractmethod
    def catalog(self) -> BaseCatalog:
        """Access the catalog interface."""
        pass

    @abstractmethod
    def stop(self, skip_usage_summary: bool) -> None:
        """Clean up the session state."""
        pass
