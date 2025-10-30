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
)
from fenic._inference.language_model import InferenceConfiguration, LanguageModel
from fenic.core._logical_plan.resolved_types import (
    ResolvedModelAlias,
    ResolvedResponseFormat,
)
from fenic.core.types import ClassifyExample, ClassifyExampleCollection

logger = logging.getLogger(__name__)


EXAMPLES = ClassifyExampleCollection()
EXAMPLES.create_example(
    ClassifyExample(
        input="I absolutely loved this product! It exceeded all my expectations and made my daily routine so much easier. "
        "The customer service team was also incredibly helpful when I had questions. I've been using it for about a month now, "
        "and it continues to perform flawlessly.",
        output="positive",
    )
)
EXAMPLES.create_example(
    ClassifyExample(
        input="We decided to try this restaurant for our anniversary dinner last Saturday. We had made reservations a week in advance, "
        "but still waited about 25 minutes to be seated. The menu looked promising, with several seasonal specialties. "
        "Our appetizer arrived after about 30 minutes, and the main courses followed 45 minutes later. "
        "The temperature of the food wasn't what we expected, and we had to ask twice for water refills. "
        "The bill came to $175 for the two of us, which was more than we had anticipated spending. "
        "On the drive home, we discussed trying a different venue for next year's celebration.",
        output="negative",
    )
)
EXAMPLES.create_example(
    ClassifyExample(
        input="The hotel was located 2 miles from downtown. It offers free parking and WiFi. Check-in is at 3pm and check-out is at 11am. "
        "The building is a 15-story structure built in 2008 with 230 rooms of various types. "
        "A continental breakfast is served daily from 6:30am to 10am in the main floor dining area. "
        "The hotel has three conference rooms available for business meetings, and a fitness center that operates from 5am to 11pm. "
        "There is a shuttle service that runs to the airport every hour on the hour.",
        output="neutral",
    )
)
EXAMPLES.create_example(
    ClassifyExample(
        input="The new update to the app has dramatically improved performance. Pages load faster, the interface is more intuitive, "
        "and I haven't experienced any crashes since installing it. I use this app daily for work and my productivity has increased "
        "significantly since the update. The developers clearly listened to user feedback and addressed the most common pain points.",
        output="positive",
    )
)
EXAMPLES.create_example(
    ClassifyExample(
        input="I regret purchasing this laptop. It constantly freezes, the battery drains within two hours, and the customer support "
        "refused to help with these issues. I've had to restart it multiple times during important meetings, which has been embarrassing "
        "and frustrating. The advertised 8-hour battery life is nowhere close to reality - I'm lucky if I get 2 hours of use before "
        "needing to find a power outlet. I've spent hours on the phone with technical support only to be told that these issues "
        "aren't covered under warranty. The money I spent on this device feels completely wasted at this point.",
        output="negative",
    )
)
EXAMPLES.create_example(
    ClassifyExample(
        input="This report summarizes the quarterly sales figures across all regions for Q1 2025. "
        "The Northeast region recorded a sales volume of 12%, showing a 2% change from the previous quarter. "
        "The Southern territory accounted for 28% of total sales, with their new product line now representing 30% of regional transactions. "
        "Meanwhile, Midwest operations contributed 25% to the company's overall performance, maintaining consistent numbers "
        "from the last three quarters.",
        output="neutral",
    )
)
EXAMPLES.create_example(
    ClassifyExample(
        input="I wasn't expecting much when I first tried this product, given my experience with similar items in the past. "
        "It took a few days to get used to the interface, which is different from what I'm familiar with. "
        "After using it for about two weeks now, I find myself reaching for it more often than my previous solution. "
        "The learning curve wasn't too steep, and I appreciate that it doesn't require constant charging like my old one did. "
        "When my colleague asked for recommendations yesterday, I mentioned this one among a couple of other options.",
        output="positive",
    )
)
EXAMPLES.create_example(
    ClassifyExample(
        input="The movie fails on almost every level. The plot has massive holes, the acting is wooden, and the special effects look cheap. "
        "I wish I could get those two hours of my life back. The director's previous work had shown such promise, "
        "making this disappointment even more acute. Characters make decisions that contradict their established motivations, "
        "and the dialogue often feels forced and unnatural.",
        output="negative",
    )
)
EXAMPLES.create_example(
    ClassifyExample(
        input="The documentary covers events from 1992 to 2001. It includes interviews with 15 participants and archival footage "
        "from various news sources. The total runtime is 90 minutes. The film is structured chronologically, "
        "with each segment dedicated to a specific year in the timeline. The director incorporated both government records "
        "and personal testimonies to reconstruct the sequence of events. Background music is minimal, "
        "mostly appearing during transitions between segments. The narration is provided by a veteran news correspondent "
        "who reported on the original events.",
        output="neutral",
    )
)

SENTIMENT_ANALYSIS_MODEL = create_classification_pydantic_model(["positive", "negative", "neutral"])
SENTIMENT_ANALYSIS_FORMAT = ResolvedResponseFormat.from_pydantic_model(SENTIMENT_ANALYSIS_MODEL, generate_struct_type=False)

class AnalyzeSentiment(BaseSingleColumnInputOperator[str, str]):
    SYSTEM_PROMPT = """You are a sentiment analysis expert.
        Your task is to determine the emotional tone expressed in the document.
        Classify each document into exactly one of these three categories:
        - "positive": Use when the text expresses approval, optimism, satisfaction, or happiness
        - "negative": Use when the text expresses disapproval, pessimism, dissatisfaction, or unhappiness
        - "neutral": Use when the text is primarily factual, balanced between positive and negative, or lacks clear emotional indicators

        Focus on the overall sentiment, even if there are mixed emotions present.
        Respond with ONLY one word: "positive", "negative", or "neutral" without any explanation or additional text.
    """

    def __init__(
        self,
        input: pl.Series,
        model: LanguageModel,
        temperature: float,
        model_alias: Optional[ResolvedModelAlias] = None,
        request_timeout: Optional[float] = None,
    ):
        super().__init__(
            input,
            CompletionOnlyRequestSender(
                model=model,
                operator_name="semantic.analyze_sentiment",
                inference_config=InferenceConfiguration(
                    max_output_tokens=MAX_TOKENS_DETERMINISTIC_OUTPUT_SIZE,
                    temperature=temperature,
                    response_format=SENTIMENT_ANALYSIS_FORMAT,
                    model_profile=model_alias.profile if model_alias else None,
                    request_timeout=request_timeout,
                ),
            ),
            EXAMPLES,
        )

    def build_system_message(self) -> str:
        return self.SYSTEM_PROMPT

    def postprocess(self, responses: List[Optional[str]]) -> List[Optional[str]]:
        predictions = []
        for response in responses:
            if not response:
                predictions.append(None)
            else:
                try:
                    data = json.loads(response)["output"]
                    predictions.append(data)
                    if data not in ["positive", "negative", "neutral"]:
                        logger.warning(
                            f"Model returned invalid label '{data}'. Valid labels: positive, negative, neutral"
                        )
                        predictions.append(None)
                except Exception as e:
                    logger.warning(
                        f"Invalid model output: {response} for semantic.analyze_sentiment: {e}"
                    )
                    predictions.append(None)
        return predictions

    def convert_example_to_assistant_message(self, example: ClassifyExample) -> str:
        return SENTIMENT_ANALYSIS_MODEL(output=example.output).model_dump_json()
