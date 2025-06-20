# noqa: D100
import logging
import os
import random
import string

from fenic.api.functions import col, text
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

s3_path = os.getenv("TEST_S3_PATH", "s3://my_bucket/integration_test_data/test_cloud_session")

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


def main(): # noqa: D103
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
                    model_name="gpt-4o-mini", rpm=1000, tpm=1000000
                )},
                default_language_model="model1",
            ),
        )
    )
    logger.info("Session created successfully")

    data = {
        "name": ["Alice", "Bob", "Charlie", "David", None, "Alice"],
        "age": [None, 30, 30, 95, 25, 20],
        "group": [100, 300, 300, 100, 200, 300],
        "city": [
            "Product with Null",
            "San Francisco",
            "Seattle",
            "Largest Product",
            "San Francisco",
            "Denver",
        ],
    }

    df = session.create_dataframe(data)
    try:
        df.write.parquet(f"{s3_path}/test_file.parquet", mode="error")
    except Exception as e:
        logger.error(f"Error writing parquet file: {e}")

    logger.info("Testing simple write to csv file")
    df = session.create_dataframe(data)
    original_schema = df.schema
    df.write.csv(f"{s3_path}/test_file.csv")

    logger.info("Testing simple infer schema from parquet file")
    df = session.read.csv(f"{s3_path}/test_file.csv")
    assert df.schema == original_schema  # nosec: B101
    df.show()

    logger.info("Testing simple write to parquet file")
    df = session.create_dataframe(data)
    original_schema = df.schema
    df.write.parquet(f"{s3_path}/test_file.parquet")

    logger.info("Testing simple infer schema from parquet file")
    df = session.read.parquet(f"{s3_path}/test_file.parquet")
    assert df.schema == original_schema  # nosec: B101
    df.show()

    logger.info("Testing simple show")
    df = session.create_dataframe(data)
    df.select("*").show()

    logger.info("Testing complex show")
    df = session.create_dataframe(data)
    df = df.select("*", text.concat(col("group") + col("age")).alias("group_age"))
    df.show()

    logger.info("Testing simple to_polars")
    df = session.create_dataframe(data)
    result = df.to_polars()
    logger.info("Collect result: %s", result)

    logger.info("Testing complex count")
    df = session.create_dataframe(data)
    df = df.select("*", text.concat(col("group") + col("age")).alias("group_age"))
    result = df.count()
    logger.info("Count result: %s", result)

    def test_write_and_infer(csv=True):
        orig_df = session.create_dataframe(data)
        if csv:
            orig_df.write.csv(f"{s3_path}/test_file.csv", mode="overwrite")
            df = session.read.csv(f"{s3_path}/test_file.csv")

            try:
                df.write.csv(f"{s3_path}/test_file.csv", mode="error")
            except Exception as e:
                logger.error(f"Error writing csv file: {e}")
        else:
            orig_df.write.parquet(f"{s3_path}/test_file.parquet", mode="overwrite")
            df = session.read.parquet(f"{s3_path}/test_file.parquet")

            try:
                df.write.parquet(f"{s3_path}/test_file.parquet", mode="error")
            except Exception as e:
                logger.error(f"Error writing parquet file: {e}")
        assert df.schema == orig_df.schema  # nosec: B101
        logger.info("Schema: %s", df.schema)
        df.show()

    logger.info("Starting batch of 100 operations")
    for i in range(100):
        df = _randomized_df(session)

        logger.info("--------------------------------")
        if i % 5 == 0:
            logger.info("Operation %d: Showing dataframe", i)
            df.select("*").show()
        elif i % 5 == 1:
            logger.info("Operation %d: Collecting dataframe", i)
            result = df.to_polars()
            logger.info("Collect result: %s", result)
        elif i % 5 == 2:
            logger.info("Operation %d: Counting dataframe", i)
            result = df.count()
            logger.info("Count result: %s", result)
        elif i % 5 == 3:
            logger.info("Operation %d: Writing and inferring csv", i)
            test_write_and_infer(csv=True)
        else:
            logger.info("Operation %d: Writing and inferring parquet", i)
            test_write_and_infer(csv=False)
        logger.info("--------------------------------")
    logger.info("Stopping session")
    # session.stop()
    logger.info("Test completed successfully")


if __name__ == "__main__":
    main()
