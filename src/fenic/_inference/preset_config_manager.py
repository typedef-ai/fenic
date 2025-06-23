"""Shared preset configuration management for model clients."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, Optional, TypeVar

PresetT = TypeVar('PresetT')
ConfigT = TypeVar('ConfigT')


@dataclass
class BasePresetConfiguration:
    pass

class PresetConfigurationManager(Generic[PresetT, ConfigT], ABC):
    """Abstract base class for managing preset configurations across providers."""
    
    def __init__(self, 
                 preset_configurations: Optional[Dict[str, PresetT]] = None,
                 default_preset_name: Optional[str] = None):
        """Initialize the preset configuration manager.
        
        Args:
            preset_configurations: Dictionary mapping preset names to configurations
            default_preset_name: Name of the default preset to use when none specified
        """
        self.preset_configurations: Dict[str, ConfigT] = {}
        self.default_preset_name = default_preset_name
        
        if preset_configurations:
            for name, preset in preset_configurations.items():
                self.preset_configurations[name] = self._process_preset(preset)
    
    @abstractmethod
    def _process_preset(self, preset: PresetT) -> ConfigT:
        """Process a raw preset configuration into the provider-specific format.
        
        Args:
            preset: Raw preset configuration from session config
            
        Returns:
            Processed configuration object for this provider
        """
        pass
    
    @abstractmethod
    def _get_default_configuration(self) -> ConfigT:
        """Get the default configuration when no preset is specified.
        
        Returns:
            Default configuration object
        """
        pass
    
    def get_preset_configuration(self, preset_name: Optional[str]) -> ConfigT:
        """Get the configuration for a given preset name.
        
        Args:
            preset_name: Name of the preset to get configuration for
            
        Returns:
            Configuration object for the preset
        """
        if preset_name is None:
            preset_name = self.default_preset_name
        if preset_name is None:
            return self._get_default_configuration()
        return self.preset_configurations.get(preset_name, self._get_default_configuration())