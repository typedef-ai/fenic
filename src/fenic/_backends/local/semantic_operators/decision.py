"""Explicit closed-set lowering through the native judgment request path."""

import json
from typing import Optional, Sequence

from fenic._backends.local.semantic_operators.base import (
    CompletionOnlyRequestSender,
    RequestSender,
)
from fenic._inference.types import LMRequestMessages
from fenic.core._logical_plan.resolved_types import ResolvedClassDefinition
from fenic.core.error import ConfigurationError
from fenic.core.types.judge import JudgeQuestion


class DecisionRequestSender(RequestSender[str]):
    """Adapt a known predicate or label set, never infer one from a text prompt."""

    def __init__(
        self,
        sender: CompletionOnlyRequestSender,
        instructions: str,
        classes: Optional[Sequence[ResolvedClassDefinition]] = None,
    ):
        self.model = sender.model
        self.config = sender.inference_config
        if self.config.temperature not in (None, 0):
            raise ConfigurationError(
                "The TypeSafe decision provider supports only temperature=0."
            )
        if self.config.model_profile is not None:
            raise ConfigurationError(
                "The TypeSafe decision provider does not support model profiles."
            )
        instructions += (
            "\nThe state is a JSON object. Evaluate only its 'input' field. "
            "The 'examples' field contains labeled input/response pairs for guidance."
        )
        self.labels = None
        if classes is None:
            self.question = JudgeQuestion.noul(
                name="decision",
                instructions=instructions,
                criteria={
                    "true": "The input's question or claim is true.",
                    "false": "The input's question or claim is false or unclear.",
                },
            )
        else:
            if not 2 <= len(classes) <= 255:
                raise ConfigurationError(
                    "The TypeSafe decision provider requires 2..255 classes."
                )
            # Stable transport IDs preserve arbitrary labels, including slug collisions.
            self.labels = {
                f"label_{index}": item.label for index, item in enumerate(classes)
            }
            self.question = JudgeQuestion.choice(
                name="decision",
                instructions=instructions,
                options={
                    f"label_{index}": json.dumps(
                        {"label": item.label, "description": item.description},
                        ensure_ascii=False,
                    )
                    for index, item in enumerate(classes)
                },
            )

    def send_requests(
        self, messages_batch: list[Optional[LMRequestMessages]]
    ) -> list[Optional[str]]:
        """Preserve input and examples, then convert a typed answer for the operator."""
        states = [
            None
            if message is None
            else json.dumps(
                {
                    "input": message.user,
                    "examples": [
                        {"input": example.user, "response": example.assistant}
                        for example in message.examples
                    ],
                },
                ensure_ascii=False,
            )
            for message in messages_batch
        ]
        responses = self.model.get_judgments(
            states,
            (self.question,),
            model_profile=self.config.model_profile,
            request_timeout=self.config.request_timeout,
        )
        outputs = []
        for response in responses:
            if response is None:
                outputs.append(None)
                continue
            answer = json.loads(response.completion)
            value = (
                self.labels[answer["decision"]]
                if self.labels is not None
                else answer["decision_p"] > 0.5
            )
            outputs.append(json.dumps({"output": value}))
        return outputs
