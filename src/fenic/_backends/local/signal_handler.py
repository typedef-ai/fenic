"""Global signal handler for interrupt handling."""

import signal
import threading
from typing import Optional


# Global state
_original_sigint_handler: Optional[callable] = None
_handler_installed = threading.Event()


def _install_signal_handler():
    """Install global SIGINT handler (thread-safe, idempotent)"""
    global _original_sigint_handler
    
    if _handler_installed.is_set():
        return
        
    if threading.current_thread() is threading.main_thread():
        _original_sigint_handler = signal.signal(signal.SIGINT, _handle_sigint)
        _handler_installed.set()


def _handle_sigint(signum, frame):
    """Global SIGINT handler - cancels all operations"""
    print("Interrupting operations...")
    
    # Cancel operations in all sessions using existing shutdown machinery
    from fenic._backends.local.manager import LocalSessionManager
    # from fenic._backends.cloud.manager import CloudSessionManager  # if applicable
    
    # For each ModelClient across all sessions:
    # 1. Call shutdown(interrupt=True) to cancel everything
    # 2. Immediately reset shutdown_event so client is ready for next operation
    LocalSessionManager().interrupt_all_operations()
    
    # Always propagate KeyboardInterrupt
    raise KeyboardInterrupt()


def ensure_signal_handler():
    """Ensure signal handler is installed - called from Session.get_or_create()"""
    if not _handler_installed.is_set():
        _install_signal_handler()