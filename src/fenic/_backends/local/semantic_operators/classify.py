import json
import logging
from typing import List, Optional

import polars as pl

from fenic._backends.local.semantic_operators.base import (
    BaseSingleColumnInputOperator,
    CompletionOnlyRequestSender,
)
from fenic._backends.local.semantic_operators.utils import (
    create_classification_pydantic_model,
)
from fenic._constants import (
    MAX_TOKENS_DETERMINISTIC_OUTPUT_SIZE,
    TOKEN_OVERHEAD_JSON,
    TOKEN_OVERHEAD_MISC,
)
from fenic._inference.language_model import InferenceConfiguration, LanguageModel
from fenic.core._logical_plan.resolved_types import (
    ResolvedClassDefinition,
    ResolvedModelAlias,
    ResolvedResponseFormat,
)
from fenic.core.types import ClassifyExample, ClassifyExampleCollection

logger = logging.getLogger(__name__)


class Classify(BaseSingleColumnInputOperator[str, str]):
    SYSTEM_PROMPT = (
        "You are a text classification expert. "
        "Classify the following document into one of the following labels:"
        "\n{classes}\n"
        "Respond with *only* the predicted label."
    )

    def __init__(
        self,
        input: pl.Series,
        classes: List[ResolvedClassDefinition],
        model: LanguageModel,
        temperature: float,
        examples: Optional[ClassifyExampleCollection] = None,
        model_alias: Optional[ResolvedModelAlias] = None,
        request_timeout: Optional[float] = None,
    ):
        self.classes = classes
        self.valid_labels = {class_def.label for class_def in classes}
        # Create output model from class labels
        labels = [class_def.label for class_def in classes]
        self.output_model = create_classification_pydantic_model(labels)
        super().__init__(
            input,
            CompletionOnlyRequestSender(
                operator_name="semantic.classify",
                model=model,
                inference_config=InferenceConfiguration(
                    max_output_tokens=self.get_max_tokens(),
                    temperature=temperature,
                    response_format=ResolvedResponseFormat.from_pydantic_model(self.output_model, generate_struct_type=False),
                    model_profile=model_alias.profile if model_alias else None,
                    request_timeout=request_timeout,
                ),
            ),
            examples,
        )

    def build_system_message(self) -> str:
        """Build system message with class descriptions."""
        class_descriptions = []
        for class_def in self.classes:
            if class_def.description:
                class_descriptions.append(f"- {class_def.label}: {class_def.description}")
            else:
                class_descriptions.append(f"- {class_def.label}")

        classes_text = "\n".join(class_descriptions)

        return self.SYSTEM_PROMPT.format(classes=classes_text)

    def postprocess(self, responses: List[Optional[str]]) -> List[Optional[str]]:
        predictions = []
        for response in responses:
            if not response:
                predictions.append(None)
            else:
                try:
                    data = json.loads(response)["output"]
                    # Validate the response is one of the valid labels
                    if data not in self.valid_labels:
                        logger.warning(
                            f"Model returned invalid label '{data}'. Valid labels: {self.valid_labels}"
                        )
                        predictions.append(None)
                    else:
                        predictions.append(data)
                except Exception as e:
                    logger.warning(
                        f"Invalid model output: {response} for semantic.classify: {e}"
                    )
                    predictions.append(None)
        return predictions

    def convert_example_to_assistant_message(self, example: ClassifyExample) -> str:
        return self.output_model(output=example.output).model_dump_json()

    def get_max_tokens(self) -> int:
        max_label_length = max(len(class_def.label) for class_def in self.classes)
        return max(MAX_TOKENS_DETERMINISTIC_OUTPUT_SIZE, TOKEN_OVERHEAD_JSON + TOKEN_OVERHEAD_MISC + max_label_length)
