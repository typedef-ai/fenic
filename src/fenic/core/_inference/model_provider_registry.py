"""Registry for model provider instances."""

from typing import Dict

from fenic.core._inference.model_provider import ModelProvider


class ModelProviderRegistry:
    """Registry for managing model provider instances."""
    
    def __init__(self):
        self._providers: Dict[str, ModelProvider] = {}
        self._initialized = False
    
    def _initialize_providers(self):
        """Initialize all available providers."""
        if self._initialized:
            return
            
        # Import here to avoid circular dependencies
        from fenic._inference.common_openai.openai_provider import openai_provider
        from fenic._inference.anthropic.anthropic_provider import anthropic_provider
        from fenic._inference.google.google_provider import (
            google_developer_provider,
            google_vertex_provider,
        )
        from fenic._inference.cohere.cohere_provider import cohere_provider
        
        self._providers = {
            "openai": openai_provider,
            "anthropic": anthropic_provider,
            "google-developer": google_developer_provider,
            "google-vertex": google_vertex_provider,
            "cohere": cohere_provider,
        }
        self._initialized = True
    
    def get_provider(self, name: str) -> ModelProvider:
        """Get a provider by name.
        
        Args:
            name: The provider name (e.g., "openai", "anthropic")
            
        Returns:
            The provider instance
            
        Raises:
            ValueError: If the provider is not found
        """
        self._initialize_providers()
        if name not in self._providers:
            raise ValueError(f"Unknown provider: {name}. Available providers: {list(self._providers.keys())}")
        return self._providers[name]
    
    def get_all_providers(self) -> Dict[str, ModelProvider]:
        """Get all registered providers."""
        self._initialize_providers()
        return self._providers.copy()


# Create singleton instance
model_provider_registry = ModelProviderRegistry()