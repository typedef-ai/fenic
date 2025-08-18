"""Shared async utilities for Fenic backend operations."""

import asyncio
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class EventLoopManager:
    """Singleton managing shared event loop for all async operations.
    
    This class provides a centralized event loop that runs on a background thread,
    allowing multiple components (ModelClient, AsyncUDF, etc.) to share the same
    async infrastructure efficiently.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the manager with no active loop."""
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self._manager_lock = threading.Lock()
        self._client_count = 0
    
    def get_or_create_loop(self) -> asyncio.AbstractEventLoop:
        """Get existing loop or create new one on background thread.
        
        Returns:
            The shared event loop instance.
        """
        with self._manager_lock:
            if self.loop is None or self.loop.is_closed():
                self._create_event_loop()
            self._client_count += 1
            return self.loop
    
    def release_loop(self):
        """Decrement client count and shutdown loop if no clients remain."""
        loop_to_shutdown = None
        thread_to_join = None
        
        with self._manager_lock:
            self._client_count -= 1
            if self._client_count <= 0 and self.loop and self.loop.is_running():
                loop_to_shutdown = self.loop
                thread_to_join = self.thread
                self.loop = None
                self.thread = None
                self._client_count = 0
        
        # Shutdown outside lock to avoid deadlock
        if loop_to_shutdown:
            self._shutdown_loop(loop_to_shutdown, thread_to_join)
    
    def _create_event_loop(self):
        """Create and start event loop on background thread.
        
        Must be called while holding _manager_lock.
        """
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self._run_event_loop,
            args=(self.loop,),
            daemon=True,
            name="EventLoopManager-Thread"
        )
        self.thread.start()
        
        # Wait for loop to start
        while not self.loop.is_running():
            pass
        
        logger.info("Created new event loop on background thread")
    
    def _run_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Run the event loop in the background thread."""
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            loop.close()
    
    def _shutdown_loop(self, loop: asyncio.AbstractEventLoop, thread: Optional[threading.Thread]):
        """Shutdown event loop and join thread."""
        try:
            # Cancel all tasks
            cancel_future = asyncio.run_coroutine_threadsafe(
                self._cancel_all_tasks(loop), loop
            )
            cancel_future.result(timeout=5)
        except Exception as e:
            logger.warning(f"Error cancelling tasks: {e}")
        
        # Stop the loop
        loop.call_soon_threadsafe(loop.stop)
        
        # Join thread
        if thread and thread.is_alive():
            thread.join(timeout=5)
            if thread.is_alive():
                logger.warning("Event loop thread did not terminate in time")
        
        logger.info("Event loop shutdown complete")
    
    async def _cancel_all_tasks(self, loop: asyncio.AbstractEventLoop):
        """Cancel all pending tasks in the loop."""
        tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
        for task in tasks:
            task.cancel()
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)