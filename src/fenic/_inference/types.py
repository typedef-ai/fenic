from dataclasses import dataclass
from typing import Dict, List, Optional

from openai.types.chat import ChatCompletionTokenLogprob
from pydantic import BaseModel


@dataclass
class FewShotExample:
    user: str
    assistant: str

@dataclass
class LMRequestMessages:
    system: str
    examples: List[FewShotExample]
    user: str

    def to_message_list(self) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.system}]

        for example in self.examples:
            messages.append({"role": "user", "content": example.user})
            messages.append({"role": "assistant", "content": example.assistant})

        messages.append({"role": "user", "content": self.user})
        return messages

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
    max_completion_tokens: int
    top_logprobs: Optional[int]
    structured_output: Optional[type[BaseModel]]
    temperature: float
    model_profile: Optional[str] = None
