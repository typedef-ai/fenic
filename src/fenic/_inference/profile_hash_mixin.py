"""Shared profile hashing helper for model clients."""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Optional

from fenic._inference.profile_manager import BaseProfileConfiguration

logger = logging.getLogger(__name__)


class ProfileHashMixin(ABC):
    """Mixin that provides deterministic hashing of profile configurations."""

    @abstractmethod
    def _resolve_profile_for_hash(
        self, profile_name: Optional[str]
    ) -> Optional[BaseProfileConfiguration]:
        """Return the provider-specific profile configuration for hashing."""

    def get_profile_hash(self, profile_name: Optional[str]) -> Optional[str]:  # noqa: D401
        """See ModelClient.get_profile_hash."""
        try:
            profile = self._resolve_profile_for_hash(profile_name)
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.debug(
                "Failed to resolve profile for hashing (profile=%s): %s",
                profile_name,
                exc,
            )
            return None

        if profile is None:
            return None

        serialized_profile = self._serialize_profile(profile)
        return hashlib.sha256(serialized_profile.encode("utf-8")).hexdigest()

    @staticmethod
    def _serialize_profile(profile: BaseProfileConfiguration) -> str:
        """Serialize profile to a JSON string with stable ordering.

        Args:
            profile: The profile to serialize.

        Returns:
            A JSON string representing the profile.
        """
        return json.dumps(asdict(profile), sort_keys=True, default=str)

