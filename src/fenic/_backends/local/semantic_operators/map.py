from textwrap import dedent
from typing import Any, Dict, Iterator, List, Optional, Union

import jinja2
import polars as pl
from pydantic import BaseModel

from fenic._backends.local.semantic_operators.base import (
    BaseMultiColumnInputOperator,
    CompletionOnlyRequestSender,
)
from fenic._backends.local.semantic_operators.extract import Extract
from fenic._backends.local.semantic_operators.utils import (
    SCHEMA_EXPLANATION_INSTRUCTION_FRAGMENT,
    SIMPLE_INSTRUCTION_SYSTEM_PROMPT,
)
from fenic._inference.language_model import InferenceConfiguration, LanguageModel
from fenic._inference.types import LMRequestMessages
from fenic.core._logical_plan.resolved_types import (
    ResolvedModelAlias,
    ResolvedResponseFormat,
)
from fenic.core._utils.schema import convert_custom_dtype_to_polars
from fenic.core.error import InternalError
from fenic.core.types import (
    MapExample,
    MapExampleCollection,
)


class MapExtract:
    """Pipe a string ``semantic.map`` result directly into ``semantic.extract``.

    This internal adapter deliberately composes the existing row-local request
    senders instead of adding another model-client API.  Pulling an extract
    request pulls only the map responses needed to fill that extract request
    block, so the full intermediate map result is never materialized.
    """

    def __init__(self, map_operator: "Map", extract_operator: "Extract"):
        self.map_operator = map_operator
        self.extract_operator = extract_operator

    def execute(self) -> pl.Series:
        if self.map_operator.request_batch_size != self.extract_operator.request_batch_size:
            raise ValueError("Fused map and extract operators must use the same request batch size")

        postprocessed_responses = []
        for response in self.extract_operator.request_sender.send_request_stream(
            self._iter_extract_request_messages(),
            batch_size=self.extract_operator.request_batch_size,
        ):
            postprocessed_responses.extend(self.extract_operator.postprocess([response]))
        return pl.Series(postprocessed_responses, dtype=self.extract_operator.output_type)

    def _iter_extract_request_messages(self) -> Iterator[Optional[LMRequestMessages]]:
        for response in self.map_operator.request_sender.send_request_stream(
            self.map_operator.iter_request_messages(),
            batch_size=self.map_operator.request_batch_size,
        ):
            mapped = self.map_operator.postprocess([response])[0]
            # Match Extract.iter_request_messages: empty map output is a null
            # extract input rather than a request with an empty user message.
            if not mapped:
                yield None
            elif not isinstance(mapped, str):
                raise InternalError("Fused semantic.map must produce string output")
            else:
                yield self.extract_operator.build_request_messages(mapped)


class Map(BaseMultiColumnInputOperator[str, str]):
    stream_requests = True
    RESPONSE_FORMAT_SYSTEM_PROMPT = jinja2.Template(
        dedent("""\
            Follow the user's instruction exactly and generate output according to the user's schema.

            Output Schema:
            {{ schema_definition }}

            {{ schema_explanation }}

            Requirements:
            1. Follow the instruction exactly as written
            2. Generate output that matches the provided schema exactly
            3. Include all required fields - no extra fields, no missing fields
            4. Each field's content must match its description precisely""").strip()
    )

    def __init__(
        self,
        input: pl.Series,
        jinja_template: str,
        model: LanguageModel,
        max_tokens: int,
        temperature: float,
        model_alias: Optional[ResolvedModelAlias] = None,
        response_format: Optional[ResolvedResponseFormat] = None,
        examples: Optional[MapExampleCollection] = None,
        request_timeout: Optional[float] = None,
    ):
        super().__init__(
            input,
            request_sender=CompletionOnlyRequestSender(
                model=model,
                operator_name="semantic.map",
                inference_config=InferenceConfiguration(
                    max_output_tokens=max_tokens,
                    response_format=response_format,
                    temperature=temperature,
                    model_profile=model_alias.profile if model_alias else None,
                    request_timeout=request_timeout,
                ),
            ),
            jinja_template=jinja2.Template(jinja_template),
            examples=examples,
            output_type=convert_custom_dtype_to_polars(response_format.struct_type) if response_format else None
        )
        self.response_format = response_format

    def build_system_message(self) -> str:
        if self.response_format is not None:
            if not self.response_format.prompt_schema_definition:
                raise InternalError("Missing prompt_schema_definition for structured response format in semantic.map")
            return self.RESPONSE_FORMAT_SYSTEM_PROMPT.render(
                schema_explanation=SCHEMA_EXPLANATION_INSTRUCTION_FRAGMENT,
                schema_definition=self.response_format.prompt_schema_definition,
            )
        else:
            return SIMPLE_INSTRUCTION_SYSTEM_PROMPT

    def postprocess(
        self, responses: List[Optional[str]]
    ) -> Union[List[Optional[Dict[str, Any]]], List[Optional[str]]]:
        if self.response_format is None:
            return responses
        return [
            self.response_format.parse_structured_response(
                json_resp, "semantic.map"
            )
            for json_resp in responses
        ]

    def convert_example_to_assistant_message(self, example: MapExample) -> str:
        """Convert a MapExample to an assistant message string.

        If the example output is a BaseModel instance, serialize it to JSON.
        Otherwise, return the string output directly.
        """
        if isinstance(example.output, BaseModel):
            return example.output.model_dump_json()
        return example.output
