# noqa: D100
import logging
import random
import string

from fenic.api.session import (
    CloudConfig,
    CloudExecutorSize,
    SemanticConfig,
    Session,
    SessionConfig,
)
from fenic.api.session.config import OpenAIModelConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _randomized_df(session, num_rows=None, num_cols=None): # noqa: D417
    """Creates a randomized DataFrame for testing.

    Args:
        local_session: The session to create the DataFrame in
        num_rows: Optional number of rows (default: random 3-10)
        num_cols: Optional number of columns (default: random 3-5)

    Returns:
        A DataFrame with random data
    """
    if num_rows is None:
        num_rows = random.randint(3, 10)  # nosec: B311
    if num_cols is None:
        num_cols = random.randint(3, 5)  # nosec: B311

    # Define possible column types
    column_types = {
        "int": lambda: random.randint(1, 100),  # nosec: B311
        "float": lambda: round(random.uniform(1.0, 100.0), 2),  # nosec: B311
        "string": lambda: "".join(
            random.choices(string.ascii_letters, k=random.randint(5, 10))  # nosec: B311
        ),
        "bool": lambda: random.choice([True, False]),  # nosec: B311
    }

    # Generate column names and types
    columns = {}
    for i in range(num_cols):
        col_type = random.choice(list(column_types.keys()))  # nosec: B311
        col_name = f"col_{i}_{col_type}"
        columns[col_name] = [column_types[col_type]() for _ in range(num_rows)]

    return session.create_dataframe(columns)


def main():
    logger.info("Starting cloud session test")

    session = Session.get_or_create(
        SessionConfig(
            app_name="test_app",
            db_path="test_db",
            cloud=CloudConfig(
                size=CloudExecutorSize.SMALL,
            ),
             semantic=SemanticConfig(
                language_models={ "model1": OpenAIModelConfig(
                    model_name="gpt-4.1-nano", rpm=500, tpm=200_000
                )},
                default_language_model="model1",
            ),
        )
    )
    logger.info("Session created successfully")

    logger.info("Starting batch of 100 operations")
    for i in range(100):
        df = _randomized_df(session)

        logger.info("--------------------------------")
        if i % 3 == 0:
            logger.info("Operation %d: Showing dataframe", i)
            df.select("*").show()
        elif i % 3 == 1:
            logger.info("Operation %d: Collecting dataframe", i)
            result = df.to_polars()
            logger.info("Collect result: %s", result)
        elif i % 3 == 2:
            logger.info("Operation %d: Counting dataframe", i)
            result = df.count()
            logger.info("Count result: %s", result)
        logger.info("--------------------------------")
    logger.info("Stopping session")
    # session.stop()
    logger.info("Test completed successfully")


if __name__ == "__main__":
    main()
