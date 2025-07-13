from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar

import polars as pl

from fenic._inference.language_model import InferenceConfiguration, LanguageModel
from fenic._inference.types import FewShotExample, LMRequestMessages
from fenic.core.types.semantic_examples import BaseExampleCollection

# Type variable for raw model output (e.g., completion text, completion text with log probabilities, explanations, etc.)
ModelResponseType = TypeVar("ModelResponseType")

# Type variable for the final output returned by the operator after post-processing
OperatorOutputType = TypeVar("OperatorOutputType")


class RequestSender(Generic[ModelResponseType], ABC):
    """Abstract interface for sending LLM prompts and receiving model responses."""

    @abstractmethod
    def send_requests(
        self,
        messages_batch: List[Optional[LMRequestMessages]],
    ) -> List[Optional[ModelResponseType]]:
        """Send a batch of requests to the language model.

        Args:
            messages_batch: A list of optional lists of LMRequestMessages. `None` prompts shouldn't be sent to the model.

        Returns:
            A list of model responses, in the same order as the input messages_batch.
        """
        pass


class CompletionOnlyRequestSender(RequestSender[str]):
    """A simple request sender for models that return only raw text completions."""

    def __init__(
        self,
        model: LanguageModel,
        operator_name: str,
        inference_config: InferenceConfiguration,
    ):
        """Initializes a request sender for models that return raw text completions.

        Args:
            model (LanguageModel): The language model used to generate completions.
            operator_name (str): Name of the operator (used for logging/progress bar).
            inference_config (InferenceConfiguration): Inference-time settings and parameters.
        """
        self.model = model
        self.operator_name = operator_name
        self.inference_config = inference_config

    def send_requests(
        self,
        message_batch: List[Optional[LMRequestMessages]],
    ) -> List[Optional[str]]:
        """Send prompts to the model and return string completions.

        Args:
            message_batch: List of Prompt objects or None.

        Returns:
            List of model-generated text outputs.
        """
        messages_list = [
           message if message is not None else None
            for message in message_batch
        ]
        responses = self.model.get_completions(
            messages=messages_list,
            operation_name=self.operator_name,
            max_tokens=self.inference_config.max_output_tokens,
            temperature=self.inference_config.temperature,
            response_format=self.inference_config.response_format,
            top_logprobs=self.inference_config.top_logprobs,
        )

        completions = [
            response.completion if response else None for response in responses
        ]
        return completions

class BaseOperator(Generic[ModelResponseType, OperatorOutputType], ABC):
    """Abstract base class for any operator that sends builds LM requests and processes responses."""

    request_sender: RequestSender[ModelResponseType]

    def __init__(self, request_sender: RequestSender[ModelResponseType]):
        """Initializes the component with a request sender.

        Args:
            request_sender (RequestSender[ModelResponseType]):
                Component responsible for sending requests and retrieving model responses.
        """
        self.request_sender = request_sender

    def execute(self) -> pl.Series:
        """Run the full request building -> model -> postprocess pipeline.

        Returns:
            A Polars Series containing the final operator output (e.g., classification, label, etc).
        """
        prompts = self.build_request_messages_batch()
        responses = self.request_sender.send_requests(prompts)
        return pl.Series(self.postprocess(responses))

    @abstractmethod
    def build_request_messages_batch(self) -> List[Optional[LMRequestMessages]]:
        """Construct the LLM prompts from the operator's input.

        Returns:
            A list of Prompts or None entries, one per row.
        """
        pass

    @abstractmethod
    def postprocess(
        self, responses: List[Optional[ModelResponseType]]
    ) -> List[Optional[OperatorOutputType]]:
        """Convert raw model responses into final operator output.

        Args:
            responses: List of model responses, possibly with None values.

        Returns:
            List of postprocessed results, aligned with input.
        """
        pass


class BaseSingleColumnInputOperator(
    BaseOperator[ModelResponseType, OperatorOutputType], ABC
):
    """Base class for operators that take a single text column as input."""

    def __init__(
        self,
        input: pl.Series,
        request_sender: RequestSender[ModelResponseType],
        examples: Optional[BaseExampleCollection],
    ):
        """Initializes the component with input data, request sender, and optional examples.

        Args:
            input (pl.Series):
                A Polars Series containing one string per row.
            request_sender (RequestSender[ModelResponseType]):
                Component responsible for sending LLM requests.
            examples (Optional[BaseExampleCollection]):
                Optional labeled examples to include in the prompt for few-shot learning.
        """
        super().__init__(request_sender)
        self.input = input
        self.examples = examples

    def build_request_messages_batch(self) -> List[LMRequestMessages | None]:
        messages_batch = []
        for document in self.input:
            if not document:
                messages_batch.append(None)
            else:
                messages_batch.append(
                    self.build_request_messages(
                        document, self.build_examples() if self.examples else []
                    )
                )
        return messages_batch

    def build_request_messages(
        self, input: str, examples: List[FewShotExample]
    ) -> LMRequestMessages:
        """Construct a prompt from a single input and optional in-context examples."""
        return LMRequestMessages(
            system=self.build_system_message(),
            user=self.build_user_message(input),
            examples=examples,
        )

    def build_examples(self) -> List[FewShotExample]:
        """Convert the stored example collection into prompt message pairs.

        Returns:
            List of (UserMessage, AssistantMessage) tuples.
        """
        formatted_examples: List[FewShotExample] = []

        for example in self.examples.examples:
            formatted_examples.append(
                FewShotExample(
                    user=self.build_user_message(example.input),
                    assistant=self.convert_example_to_assistant_message(example),
                )
            )

        return formatted_examples

    def convert_example_to_assistant_message(self, example) -> str:
        """Extract the assistant message text from a labeled example."""
        return example.output

    @abstractmethod
    def build_system_message(self) -> str:
        """Construct the system message for the prompt.
        This typically includes task instructions or context.

        Returns:
            A string containing the system message content.
        """
        pass

    def build_user_message(self, input: str) -> str:
        """Construct the user message portion of the prompt from a string input.

        Args:
            input: Single input text string.

        Returns:
            A string containing the user message content.
        """
        return input


class BaseMultiColumnInputOperator(
    BaseOperator[ModelResponseType, OperatorOutputType], ABC
):
    """Base class for operators that consume multiple input fields per row."""

    def __init__(
        self,
        input: pl.DataFrame,
        request_sender: RequestSender[ModelResponseType],
        examples: Optional[BaseExampleCollection],
    ):
        """Initializes the request component with input data, a request sender, and optional examples.

        Args:
            input (pl.DataFrame):
                A Polars DataFrame containing multiple input columns.
            request_sender (RequestSender[ModelResponseType]):
                Component responsible for sending LLM requests.
            examples (Optional[BaseExampleCollection]):
                Optional labeled examples to include for few-shot prompting.
        """
        super().__init__(request_sender)
        self.input = input
        self.examples = examples

    def build_request_messages_batch(self) -> List[LMRequestMessages | None]:
        messages_batch = []
        for input in self.input.iter_rows(named=True):
            if not input or not all(input.values()):
                messages_batch.append(None)
            else:
                messages_batch.append(
                    self.build_request_messages(
                        input, self.build_examples() if self.examples else []
                    )
                )
        return messages_batch

    def build_request_messages(
        self,
        input: dict[str, str],
        formatted_examples: List[FewShotExample],
    ) -> LMRequestMessages:
        """Construct a prompt from a dict of column values and in-context examples."""
        return LMRequestMessages(
            system=self.build_system_message(),
            user=self.build_user_message(input),
            examples=formatted_examples,
        )

    def build_examples(self) -> List[FewShotExample]:
        """Convert example input/output pairs to message format for in-context learning.

        Returns:
            List of (UserMessage, AssistantMessage) tuples.
        """
        formatted_examples = []
        for example in self.examples.examples:
            formatted_examples.append(
                FewShotExample(
                    user=self.build_user_message(example.input),
                    assistant=self.convert_example_to_assistant_message(example),
                )
            )

        return formatted_examples

    def convert_example_to_assistant_message(self, example):
        """Extract assistant message text from a labeled example."""
        return example.output

    @abstractmethod
    def build_system_message(self) -> str:
        """Create the system message for a multi-column input row.

        Returns:
            A string containing the system message content.
        """
        pass

    @abstractmethod
    def build_user_message(self, input: dict[str, str]) -> str:
        """Construct the user message from a row of named input values.

        Args:
            input: Dictionary mapping column names to string values.

        Returns:
            A string containing the user message content.
        """
        pass
