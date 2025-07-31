"""This is a MCP server for the Fenic project.

It is used to search the Fenic codebase and provide documentation for the Fenic API.
"""
import datetime
import logging
import os
import threading
import uuid
from typing import List, Literal

from fastmcp import FastMCP
from utils.schemas import get_learnings_schema
from utils.tree_operations import build_tree, tree_to_string

import fenic as fc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FenicSession:
    """Singleton class to manage Fenic session."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, "_session"):
            try:
                self._session = self._create_session()
            except Exception as e:
                logger.error(f"Error creating Fenic session: {e}")
                raise
    
    def get_session(self):
        """Get the Fenic session."""
        return self._session

    @classmethod
    def _create_session(cls):
        work_dir = os.environ.get("FENIC_WORK_DIR", os.path.expanduser("~/.fenic"))
        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)

        # Set DuckDB temp directory
        os.environ["DUCKDB_TMPDIR"] = work_dir
            
        # Check if required API keys are available
        config = cls._get_config()

        return fc.Session.get_or_create(config)
    
    @classmethod
    def _get_config(cls) -> fc.SessionConfig:
        if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
            logger.warning("No API keys found. Set OPENAI_API_KEY and/or GEMINI_API_KEY environment variables.")
            # Create minimal config for now
            config = fc.SessionConfig(app_name="docs")
        else:
            config = fc.SessionConfig(
                app_name="docs",
                semantic=fc.SemanticConfig(
                    language_models={
                        "flash": cls._get_google_config("gemini-2.0-flash"),
                        "flash-lite": cls._get_google_config("gemini-2.0-flash-lite"),
                        "mini": cls._get_openai_config("gpt-4.1-mini"),
                        "nano": cls._get_openai_config("gpt-4.1-nano"),
                    },
                    default_language_model="flash",
                    embedding_models={
                        "large": cls._get_openai_config("text-embedding-3-large", 3_000, 1_000_000)
                    }
                ),
            )
        return config

    @classmethod
    def _get_google_config(
            cls,
            model_name: str,
            rpm: int = 2_000,
            tpm: int = 4_000_000) -> fc.GoogleGLAModelConfig:
        return fc.GoogleGLAModelConfig(
            model_name=model_name,
            rpm=rpm,
            tpm=tpm
        )

    @classmethod
    def _get_openai_config(
            cls,
            model_name: str,
            rpm: int = 500,
            tpm: int = 200_000) -> fc.OpenAIModelConfig:
        return fc.OpenAIModelConfig(
            model_name=model_name,
            rpm=rpm,
            tpm=tpm
        )


def initialize_learnings_table(include_embeddings: bool = True) -> bool:
    """Initialize the learnings table if it doesn't exist.
    
    Note: Fenic fully supports ArrayType in table schemas. The limitation about "primitive types only" 
    applies specifically to CSV import schemas, not table schemas in general.
    
    Args:
        include_embeddings: Whether to include embedding columns in the schema
        
    Returns:
        bool: True if table was created, False if it already existed
    """
    table_name = "learnings"
    session = FenicSession().get_session()
    # Check if table already exists
    if session.catalog.does_table_exist(table_name):
        return False
    
    # Get schema from utility module
    learnings_schema = get_learnings_schema(include_embeddings)
    
    # Create the table
    session.catalog.create_table(table_name, learnings_schema)
    return True

mcp = FastMCP("Fenic Documentation")

@mcp.tool()
def search(query: str, max_results: int = 30) -> str:
      """Search the Fenic codebase for functions, classes, methods, and other code elements.

      This tool is used to search the Fenic codebase for functions, classes, methods, and other code elements.
      It also searches through stored learnings from previous interactions.

      Also searches through stored learnings from previous interactions.

      Args:
          query: Search term or regex pattern to find in code names, documentation, and signatures
          max_results: Maximum number of results to return (default: 30)

      Returns:
          Search results with type, name, qualified path, and brief description

      Examples:
          - Simple search: "join"
          - Regex search: "semantic.*extract"
          - Search for specific terms: "DataFrame"
      """
      try:
          session = FenicSession.get_session()
          
          # Search learnings first
          learnings_results = None
          learnings_count = 0
          if session.catalog.does_table_exist("learnings"):
              try:
                  learnings_df = session.table("learnings")
                  
                  # Search across question, answer, and keywords using regex
                  learnings_search = learnings_df.filter(
                      fc.col("question").rlike(f"(?i){query}") |
                      fc.col("answer").rlike(f"(?i){query}") |
                      fc.array_contains(fc.col("keywords"), query)
                  )
                  
                  # Add relevance scoring for learnings
                  learnings_scored = learnings_search.select(
                      "question", "answer", "learning_type", "keywords", "related_functions",
                      fc.when(fc.col("question").rlike(f"(?i){query}"), fc.lit(10)).otherwise(fc.lit(0)).alias("question_score"),
                      fc.when(fc.col("answer").rlike(f"(?i){query}"), fc.lit(5)).otherwise(fc.lit(0)).alias("answer_score"),
                      fc.when(fc.array_contains(fc.col("keywords"), query), fc.lit(3)).otherwise(fc.lit(0)).alias("keywords_score")
                  )
                  
                  # Calculate total score with correction boost
                  learnings_scored = learnings_scored.select(
                      "*",
                      (fc.col("question_score") + fc.col("answer_score") + fc.col("keywords_score")).alias("base_score")
                  ).select(
                      "*",
                      fc.when(fc.col("learning_type") == "correction", 
                             fc.col("base_score") * 1.5).otherwise(fc.col("base_score")).alias("score")
                  )
                  
                  # Sort and limit learnings (max 7 results)
                  learnings_sorted = learnings_scored.order_by(fc.col("score").desc()).limit(7)
                  learnings_results = learnings_sorted.to_pydict()
                  learnings_count = len(learnings_results.get('question', []))
                  
              except Exception as e:
                  # If learnings search fails, continue with API search
                  import sys
                  print(f"Warning: Learnings search failed: {e}", file=sys.stderr)
          
          # Search API documentation
          df = session.table("api_df")

          # Filter only public API elements
          df = df.filter(
              (fc.col("is_public")) &
              (~fc.col("qualified_name").contains("._"))
          )

          # Search across all text fields
          search_df = df.filter(
              fc.col("name").rlike(f"(?i){query}") |
              fc.col("qualified_name").rlike(f"(?i){query}") |
              (fc.col("docstring").is_not_null() & fc.col("docstring").rlike(f"(?i){query}")) |
              (fc.col("annotation").is_not_null() & fc.col("annotation").rlike(f"(?i){query}")) |
              (fc.col("returns").is_not_null() & fc.col("returns").rlike(f"(?i){query}"))
          )

          # Add relevance scoring
          search_df = search_df.select(
              "type", "name", "qualified_name", "docstring",
              fc.when(fc.col("name").rlike(f"(?i){query}"), fc.lit(10)).otherwise(fc.lit(0)).alias("name_score"),
              fc.when(fc.col("qualified_name").rlike(f"(?i){query}"), fc.lit(5)).otherwise(fc.lit(0)).alias("path_score")
          )

          # Calculate total score and sort
          search_df = search_df.select(
              "*",
              (fc.col("name_score") + fc.col("path_score")).alias("score")
          )
          
          # Adjust API results limit based on learnings found
          api_limit = max(10, max_results - learnings_count)
          search_df = search_df.order_by([fc.col("score").desc(), fc.col("type"), fc.col("name")]).limit(api_limit)

          # Collect API results
          api_results = search_df.to_pydict()

          # Format output
          total_results = learnings_count + len(api_results.get('name', []))
          output = f"# Search Results for: `{query}`\n\n"
          output += f"Found {total_results} matches\n\n"

          if total_results == 0:
              output += "No results found. Try:\n"
              output += "- Different keywords (e.g., 'extract', 'semantic', 'DataFrame')\n"
              output += "- Regex patterns (e.g., 'join.*semantic')\n"
              return output

          # Show learnings first if any
          if learnings_results and learnings_count > 0:
              output += "## 📚 Learned Solutions\n\n"
              
              for i in range(learnings_count):
                  learning_type = learnings_results['learning_type'][i]
                  if learning_type == "correction":
                      output += f"### ⚠️ Correction: {learnings_results['question'][i]}\n"
                  else:
                      output += f"### 💡 {learnings_results['question'][i]}\n"
                  
                  output += f"{learnings_results['answer'][i]}\n"
                  
                  # Add metadata if available
                  if learnings_results.get('keywords') and learnings_results['keywords'][i] and len(learnings_results['keywords'][i]) > 0:
                      output += f"\n**Keywords**: {', '.join(learnings_results['keywords'][i])}\n"
                  if learnings_results.get('related_functions') and learnings_results['related_functions'][i] and len(learnings_results['related_functions'][i]) > 0:
                      output += f"**Related Functions**: {', '.join(learnings_results['related_functions'][i])}\n"
                  output += "\n---\n\n"

          # Show API results if any
          if len(api_results.get('name', [])) > 0:
              output += "## 📖 API Documentation\n"

              # Group by type for clarity
              current_type = None
              for i in range(len(api_results['name'])):
                  if api_results['type'][i] != current_type:
                      current_type = api_results['type'][i]
                      output += f"\n### {current_type.capitalize()}s\n"

                  # Format each result concisely
                  output += f"\n**`{api_results['name'][i]}`** - `{api_results['qualified_name'][i]}`\n"

                  # Add docstring if available
                  if api_results.get('docstring') and api_results['docstring'][i]:
                      output += f"  {api_results['docstring'][i]}\n"

          return output

      except Exception as e:
          return f"Search error: {str(e)}"

@mcp.tool()
def get_project_overview() -> str:
    """Get a high-level overview of the Fenic project. This should be the starting point for figuring out where to look next for specific questions."""
    session = FenicSession().get_session()
    overview = session.table("fenic_summary").select("project_summary").to_pydict()["project_summary"]
    structure = session.table("hierarchy_df").filter((fc.col("is_public")) & (fc.col("type") != "attribute") & (~fc.col("name").starts_with("_"))).select("qualified_name", "name", "type", "depth", "path_parts").to_pydict()
    tree = tree_to_string(build_tree(structure))
    result = f"## Fenic Project Overview\n\n{overview}\n\n## Fenic API Tree\n\n{tree}"
    return result

@mcp.tool()
def get_started() -> str:
    """How to get started with Fenic."""
    #session = FenicSession().get_session()
    getting_started_guide_path = os.path.join(os.path.dirname(__file__), 'fenic_getting_started_guide.md')
    with open(getting_started_guide_path, 'r') as f:
        start_guide = f.read()
    result = f"## Fenic Get Started\n\n{start_guide}"
    return result

@mcp.tool()
def get_api_tree() -> str:
    """Get the API tree of the Fenic project."""
    session = FenicSession().get_session()
    structure = session.table("hierarchy_df").filter((fc.col("is_public")) & (fc.col("type") != "attribute") & (~fc.col("name").starts_with("_"))).select("qualified_name", "name", "type", "depth", "path_parts").to_pydict()
    tree = tree_to_string(build_tree(structure))
    result = f"## Fenic API Tree\n\n{tree}"
    return result

@mcp.tool()
def store_learning(
    question: str,
    answer: str,
    learning_type: Literal["solution", "correction", "example"] = "solution",
    keywords: List[str] = None,
    related_functions: List[str] = None
) -> str:
    """Store a learning from a user interaction for future reference.
    
    WHEN TO USE THIS TOOL:
        1. After user confirms "that's correct" or "that works" following a complex solution
        2. When user corrects a mistake: "Actually, you need to..." or "That's wrong, the right way is..."
        3. After providing a multi-step solution involving 3+ Fenic operations
        4. When discovering non-obvious answers that required multiple searches
        5. When user explicitly says "remember this" or "save this for next time"

    DO NOT STORE:
        - Simple single-function lookups (e.g., "what does df.select do?")
        - Information already in basic documentation
        - Temporary debugging steps
        - User-specific data or examples

    BEST PRACTICES:
        - For corrections, use learning_type="correction" and include both wrong and right approaches
        - Extract keywords from both question and answer for better retrieval
        - Include all Fenic functions mentioned in qualified form (e.g., "DataFrame.select", "semantic.extract")
        - Keep answers concise but complete - include code examples
    
    Args:
        question: The original question or problem
        answer: The correct answer or solution
        learning_type: Type of learning (solution/correction/example)
        keywords: Search keywords for retrieval
        related_functions: Related Fenic functions (e.g., ["semantic.extract", "DataFrame.select"])
        
    Returns:
        str: The ID of the stored learning entry
    """
    session = FenicSession().get_session()
    # Initialize table if it doesn't exist
    initialize_learnings_table()
    
    # Generate unique ID and timestamp
    learning_id = str(uuid.uuid4())
    created_at = datetime.datetime.now().isoformat()
    
    # Convert None to empty lists for proper array handling
    keywords_list = keywords if keywords is not None else []
    related_functions_list = related_functions if related_functions is not None else []
    
    # Create DataFrame with the learning data (using proper arrays)
    learning_data = session.create_dataframe([{
        "id": learning_id,
        "question": question,
        "answer": answer,
        "learning_type": learning_type,
        "keywords": keywords_list,  # Store as actual array
        "related_functions": related_functions_list,  # Store as actual array
        "created_at": created_at
    }])
    
    # Add embeddings for semantic search
    learning_with_embeddings = learning_data.select(
        fc.col("id"),
        fc.col("question"),
        fc.col("answer"),
        fc.col("learning_type"),
        fc.col("keywords"),
        fc.col("related_functions"),
        fc.col("created_at"),
        fc.semantic.embed(fc.col("question")).alias("question_embedding"),
        fc.semantic.embed(fc.col("answer")).alias("answer_embedding"),
        # Create combined embedding for better search
        fc.semantic.embed(
            fc.text.concat(
                fc.col("question"), 
                fc.lit(" "), 
                fc.col("answer"), 
                fc.lit(" "), 
                fc.text.array_join(fc.col("keywords"), " ")
            )
        ).alias("combined_embedding")
    )
    
    # Store in the learnings table
    learning_with_embeddings.write.save_as_table("learnings", mode="append")
    
    return learning_id

def main():
    """Main entry point for the MCP server."""
    try:
        session = FenicSession().get_session()
        # Check if required tables exist
        required_tables = ["api_df", "hierarchy_df", "fenic_summary"]
        missing_tables = []
        for table in required_tables:
            if not session.catalog.does_table_exist(table):
                missing_tables.append(table)
        
        if missing_tables:
            logger.error(
                f"Missing required tables: {missing_tables}\n"
                "Please run 'python populate_tables.py' to set up the documentation database.\n"
                "This will extract and index the Fenic API documentation.")
            import sys
            sys.exit(1)

        mcp.run()

    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        logger.debug("Detailed error traceback:", exc_info=True)
        raise

if __name__ == "__main__":
    main()