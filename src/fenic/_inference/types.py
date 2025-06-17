from dataclasses import dataclass
from typing import Dict, List


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
