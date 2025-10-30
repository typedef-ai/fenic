import re

# Example constants
EXAMPLE_INPUT_KEY = "input"
EXAMPLE_OUTPUT_KEY = "output"
LEFT_ON_KEY = "left_on"
RIGHT_ON_KEY = "right_on"

# Indexing constants
INDEX_DIR = "indexes"
VECTOR_INDEX_DIR = f"{INDEX_DIR}/vector"

# Token count estimation constants
TOKEN_OVERHEAD_JSON = 5
TOKEN_OVERHEAD_MISC = 5
PREFIX_TOKENS_PER_MESSAGE = 4
TOKENS_PER_NAME = 1

# Default Inference Configurations
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE: float = 0

# Model client timeout constants
MAX_MODEL_CLIENT_TIMEOUT = 600  # 10 minutes
DEFAULT_MODEL_CLIENT_TIMEOUT = 120  # 2 minutes

# If the output type is known to us before runtime, this is a rough upper bound.
# Higher than expected because Anthropic estimates output tokens differently than other providers
# during tool use.
MAX_TOKENS_DETERMINISTIC_OUTPUT_SIZE = 64

API_KEY_SUFFIX = "_API_KEY"

SQL_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")

MILLISECOND_IN_SECONDS = 0.001
MINUTE_IN_SECONDS = 60
PRETTY_PRINT_INDENT = "  "
