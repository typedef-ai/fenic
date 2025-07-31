"""Shared profile configuration management for model clients."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, Optional, TypeVar

ProfileT = TypeVar("ProfileT")
ConfigT = TypeVar("ConfigT")


@dataclass
class BaseProfileConfiguration:
    pass


class ProfileManager(Generic[ProfileT, ConfigT], ABC):
    """Abstract base class for managing profile configurations across providers."""

    def __init__(
        self,
        profile_configurations: Optional[Dict[str, ProfileT]] = None,
        default_profile_name: Optional[str] = None,
    ):
        """Initialize the profile configuration manager.

        Args:
            profile_configurations: Dictionary mapping profile names to configurations
            default_profile_name: Name of the default profile to use when none specified
        """
        self.profile_configurations: Dict[str, ConfigT] = {}
        self.default_profile_name = default_profile_name

        if profile_configurations:
            for name, profile in profile_configurations.items():
                self.profile_configurations[name] = self._process_profile(profile)

    @abstractmethod
    def _process_profile(self, profile: ProfileT) -> ConfigT:
        """Process a raw profile configuration into the provider-specific format.

        Args:
            profile: Raw profile configuration from session config

        Returns:
            Processed configuration object for this provider
        """
        pass

    @abstractmethod
    def get_default_profile(self) -> ConfigT:
        """Get the default configuration when no profile is specified.

        Returns:
            Default configuration object
        """
        pass

    def get_profile_by_name(self, profile_name: Optional[str]) -> ConfigT:
        """Get the configuration for a given profile name.

        Args:
            profile_name: Name of the profile to get configuration for

        Returns:
            Configuration object for the profile
        """
        if profile_name is None:
            profile_name = self.default_profile_name
        if profile_name is None:
            return self.get_default_profile()
        return self.profile_configurations.get(profile_name, self.get_default_profile())
