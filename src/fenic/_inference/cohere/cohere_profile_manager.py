from dataclasses import dataclass
from typing import Optional

from fenic._inference.profile_manager import (
    BaseProfileConfiguration,
    ProfileManager,
)
from fenic.core._inference.model_catalog import EmbeddingModelParameters
from fenic.core._resolved_session_config import ResolvedCohereModelProfile


@dataclass
class CohereEmbeddingsProfileConfiguration(BaseProfileConfiguration):
    """Configuration for Cohere embeddings model profiles.
    
    Attributes:
        output_dimensionality: The desired output dimensionality for embeddings
        input_type: The type of input text (search_query, search_document, classification, clustering)
    
    Note: 
        Cohere supports other embedding types, but we only support float embeddings.
    """
    output_dimensionality: Optional[int] = None
    input_type: str = "search_document"

class CohereEmbeddingsProfileManager(ProfileManager[ResolvedCohereModelProfile, CohereEmbeddingsProfileConfiguration]):
    """Manages Cohere-specific profile configurations for embeddings."""

    def __init__(
        self,
        model_parameters: EmbeddingModelParameters,
        profile_configurations: Optional[dict[str, ResolvedCohereModelProfile]] = None,
        default_profile_name: Optional[str] = None,
    ):
        self.model_parameters = model_parameters
        super().__init__(profile_configurations, default_profile_name)

    def _process_profile(self, profile: ResolvedCohereModelProfile) -> CohereEmbeddingsProfileConfiguration:
        """Process Cohere profile configuration.
        
        Args:
            name: Name of the profile
            profile: The profile configuration to process
            
        Returns:
            Processed Cohere-specific profile configuration
            
        Raises:
            ConfigurationError: If dimensionality is invalid
        """
        output_dimensionality = profile.embedding_dimensionality if profile.embedding_dimensionality else None
        input_type = profile.embedding_task_type if profile.embedding_task_type else "search_document"

        return CohereEmbeddingsProfileConfiguration(
            output_dimensionality=output_dimensionality,
            input_type=input_type,
        )

    def get_default_profile(self) -> CohereEmbeddingsProfileConfiguration:
        """Get default Cohere configuration."""
        return CohereEmbeddingsProfileConfiguration()