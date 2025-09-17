import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

from openai.types.chat import ChatCompletionTokenLogprob

from fenic.core._logical_plan.resolved_types import ResolvedResponseFormat


@dataclass
class FewShotExample:
    user: str
    assistant: str

@dataclass
class LMRequestFile:
    path: str
    page_range: Tuple[int, int]
    pdf_chunk_bytes: Optional[bytes] = None
    def encode(self) -> bytes:
        data = {
            "path": self.path,
            "page_range": self.page_range
        }
        return json.dumps(data, sort_keys=True).encode('utf-8')

@dataclass
class LMRequestMessages:
    system: str
    examples: List[FewShotExample]
    user: Optional[str] = None
    user_file: Optional[LMRequestFile] = None

    def encode(self) -> bytes:
        # Convert examples to a serializable format
        examples_data = [{"user": ex.user, "assistant": ex.assistant} for ex in self.examples]
        data = {
            "system": self.system,
            "examples": examples_data,
            "user": self.user,
        }
        if self.user_file:
            data["user_file_path"] = self.user_file.path
            data["user_file_page_range"] = self.user_file.page_range
        return json.dumps(data, sort_keys=True).encode('utf-8')


@dataclass
class ResponseUsage:
    """Token usage information from API response."""
    prompt_tokens: int
    completion_tokens: int  # Actual completion tokens (non-thinking)
    total_tokens: int
    cached_tokens: int = 0
    thinking_tokens: int = 0  # Separate thinking token count

@dataclass
class FenicCompletionsResponse:
    completion: str
    logprobs: Optional[List[ChatCompletionTokenLogprob]]
    usage: Optional[ResponseUsage] = None


@dataclass
class FenicCompletionsRequest:
    messages: LMRequestMessages
    max_completion_tokens: Optional[int]
    top_logprobs: Optional[int]
    structured_output: Optional[ResolvedResponseFormat]  # Resolved JSON schema
    temperature: Optional[float]
    model_profile: Optional[str] = None

@dataclass
class FenicEmbeddingsRequest:
    doc: str
    model_profile: Optional[str] = None
