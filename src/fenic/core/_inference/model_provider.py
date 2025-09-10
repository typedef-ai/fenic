from abc import ABC, abstractmethod


class ModelProviderClass(ABC):
    """Abstract base class for model providers.
    
    This class defines the interface that all model providers must implement,
    including API key validation functionality.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the string identifier for this provider."""
        pass
    
    @abstractmethod
    def create_client(self):
        """Create a synchronous client for this provider.
        
        Returns the provider-specific synchronous client instance configured with the
        appropriate credentials and settings.
        """
        pass

    @abstractmethod
    def create_aio_client(self):
        """Create an async client for this provider.

        Returns the provider-specific asynchronous client instance configured with the
        appropriate credentials and settings.
        """
        pass
    
    @abstractmethod
    async def validate_api_key(self) -> None:
        """Validate the API key for this provider.
        
        This method should create the appropriate client and perform a lightweight 
        API call to verify that the configured credentials are valid and the 
        service is accessible. Should log success but let exceptions bubble up.
        """
        pass
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __hash__(self) -> int:
        return hash(self.name)