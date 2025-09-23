"""Ollama model management utilities for dynamic model discovery and configuration."""

import logging
from typing import Dict, List, Optional, Tuple, Any
import httpx
import ollama

logger = logging.getLogger(__name__)


class OllamaModelInfo:
    """Container for Ollama model metadata."""

    def __init__(self, model_data: Dict[str, Any]):
        self.name = model_data.get("name", "")
        self.details = model_data.get("details", {})
        self.parameters = model_data.get("parameters", {})

        # Extract key parameters
        self.context_length = self._extract_context_length()
        self.parameter_count = self._extract_parameter_count()
        self.architecture = self.details.get("family", "unknown")
        self.quantization = self._extract_quantization()

        # Determine model capabilities
        self.is_embedding_model = self._is_embedding_model()
        self.is_chat_model = self._is_chat_model()

    def _extract_context_length(self) -> int:
        """Extract context length from model parameters."""
        # Try different parameter names that might contain context length
        for key in ["general.context_length", "llama.context_length", "context_length"]:
            if key in self.parameters:
                return int(self.parameters[key])

        # Fallback defaults based on architecture/name
        if "embed" in self.name.lower():
            return 8192  # Common for embedding models
        elif any(size in self.name.lower() for size in ["7b", "8b"]):
            return 32768  # Common for 7B-8B models
        elif any(size in self.name.lower() for size in ["13b", "14b"]):
            return 16384  # Common for 13B-14B models
        elif any(size in self.name.lower() for size in ["30b", "34b"]):
            return 8192   # Common for 30B+ models
        else:
            return 8192   # Conservative default

    def _extract_parameter_count(self) -> str:
        """Extract parameter count information."""
        param_size = self.details.get("parameter_size", "")
        if param_size:
            return param_size

        # Try to infer from model name
        name_lower = self.name.lower()
        for size in ["70b", "34b", "30b", "13b", "8b", "7b", "4b", "3b", "1b"]:
            if size in name_lower:
                return size

        return "unknown"

    def _extract_quantization(self) -> str:
        """Extract quantization information."""
        # Look for quantization in model name or details
        name_lower = self.name.lower()
        for quant in ["q8_0", "q6_k", "q5_k_m", "q5_0", "q4_k_m", "q4_0", "q3_k_m", "q2_k", "fp16", "fp32"]:
            if quant in name_lower:
                return quant

        return self.details.get("quantization_level", "unknown")

    def _is_embedding_model(self) -> bool:
        """Determine if this is an embedding model based on name and metadata."""
        name_lower = self.name.lower()
        embedding_indicators = [
            "embed", "embedding", "nomic", "bge", "gte", "e5", "sentence"
        ]
        return any(indicator in name_lower for indicator in embedding_indicators)

    def _is_chat_model(self) -> bool:
        """Determine if this is a chat/completion model."""
        # Most models are chat models unless they're specifically embedding models
        return not self.is_embedding_model

    def get_max_output_tokens(self) -> int:
        """Get recommended max output tokens based on context length."""
        # Reserve some context for input, return reasonable output limit
        if self.context_length >= 32768:
            return 8192
        elif self.context_length >= 16384:
            return 4096
        elif self.context_length >= 8192:
            return 2048
        else:
            return 1024


class OllamaModelManager:
    """Manager for Ollama model discovery and metadata retrieval."""

    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self._model_cache: Dict[str, OllamaModelInfo] = {}

    async def list_available_models(self) -> List[str]:
        """Get list of locally available models using /api/tags."""
        try:
            client = ollama.AsyncClient(host=self.host)
            response = await client.list()
            models = []
            if hasattr(response, 'models') and response.models:
                for model in response.models:
                    models.append(model.model)  # Use .model attribute, not ["name"]
            return models
        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {e}")
            return []

    async def get_model_info(self, model_name: str) -> Optional[OllamaModelInfo]:
        """Get detailed model information using /api/show."""
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        try:
            # Use raw HTTP request for /api/show since ollama-python might not expose all metadata
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.host}/api/show",
                    json={"name": model_name, "verbose": True},
                    timeout=30.0
                )

                if response.status_code == 200:
                    model_data = response.json()
                    # Ensure the model name is included in the data
                    model_data["name"] = model_name
                    model_info = OllamaModelInfo(model_data)
                    self._model_cache[model_name] = model_info
                    return model_info
                elif response.status_code == 404:
                    logger.warning(f"Model '{model_name}' not found in Ollama")
                    return None
                else:
                    logger.warning(f"Failed to get model info for '{model_name}': {response.status_code}")
                    return None

        except Exception as e:
            logger.warning(f"Error getting model info for '{model_name}': {e}")
            return None

    async def discover_models_by_type(self) -> Tuple[List[str], List[str]]:
        """Discover and categorize models into chat and embedding types."""
        models = await self.list_available_models()
        chat_models = []
        embedding_models = []

        for model_name in models:
            model_info = await self.get_model_info(model_name)
            if model_info:
                if model_info.is_embedding_model:
                    embedding_models.append(model_name)
                elif model_info.is_chat_model:
                    chat_models.append(model_name)

        return chat_models, embedding_models

    async def ensure_model_available(self, model_name: str) -> bool:
        """Check if model is available, attempt to pull if not found."""
        models = await self.list_available_models()
        if model_name in models:
            return True

        logger.info(f"Model '{model_name}' not found locally, attempting to pull...")
        try:
            client = ollama.AsyncClient(host=self.host)
            await client.pull(model_name)
            logger.info(f"Successfully pulled model '{model_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to pull model '{model_name}': {e}")
            return False