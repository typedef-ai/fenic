from fastmcp import FastMCP
import os
import fenic as fc
import uuid
import datetime
from typing import List

session = None
os.chdir("/tmp")

def get_session():
    global session
    os.environ["DUCKDB_TMPDIR"] = "/tmp"
    if not session:
        config = fc.SessionConfig(
        app_name="docs",
        semantic=fc.SemanticConfig(
            language_models={
                "flash": fc.GoogleGLAModelConfig(
                    model_name="gemini-2.0-flash",
                    rpm=2000,
                    tpm=4_000_000,
                ),
                "flash-lite": fc.GoogleGLAModelConfig(
                    model_name="gemini-2.0-flash-lite",
                    rpm=4000,
                    tpm=4_000_000,
                ),
                "mini": fc.OpenAIModelConfig(
                    model_name="gpt-4.1-mini",
                    rpm=500,
                    tpm=200_000,
                ),
                "nano": fc.OpenAIModelConfig(
                    model_name="gpt-4.1-nano",
                    rpm=500,
                    tpm=200_000,
                ),
            },
            default_language_model="flash",
            embedding_models={
                "large": fc.OpenAIModelConfig(
                    model_name="text-embedding-3-large",
                    rpm=3000,
                    tpm=1_000_000
                )
            }
        ),
    )
        session = fc.Session.get_or_create(config)
    return session

def initialize_learnings_table(include_embeddings: bool = True) -> bool:
    """
    Initialize the learnings table if it doesn't exist.
    
    Note: Fenic fully supports ArrayType in table schemas. The limitation about "primitive types only" 
    applies specifically to CSV import schemas, not table schemas in general.
    
    Args:
        session: Fenic session object
        include_embeddings: Whether to include embedding columns in the schema
        
    Returns:
        bool: True if table was created, False if it already existed
    """
    table_name = "learnings"
    session = get_session()
    # Check if table already exists
    if session.catalog.does_table_exist(table_name):
        return False
    
    # Define the base learnings table schema
    schema_fields = [
        fc.ColumnField('id', fc.StringType),
        fc.ColumnField('question', fc.StringType),
        fc.ColumnField('answer', fc.StringType),
        fc.ColumnField('learning_type', fc.StringType),
        fc.ColumnField('keywords', fc.ArrayType(fc.StringType)),  # Proper array type
        fc.ColumnField('related_functions', fc.ArrayType(fc.StringType)),  # Proper array type
        fc.ColumnField('created_at', fc.StringType)
    ]
    
    # Add embedding columns if requested
    if include_embeddings:
        # Note: You may need to adjust dimensions and model based on your embedding configuration
        # The dimensions shown here (3072) match the error message for text-embedding-3-large
        embedding_type = fc.EmbeddingType(dimensions=3072, embedding_model="openai/text-embedding-3-large")
        schema_fields.extend([
            fc.ColumnField('question_embedding', embedding_type),
            fc.ColumnField('answer_embedding', embedding_type),
            fc.ColumnField('combined_embedding', embedding_type)
        ])
    
    learnings_schema = fc.Schema(schema_fields)
    
    # Create the table
    session.catalog.create_table(table_name, learnings_schema)
    return True

mcp = FastMCP("Fenic Documentation")

def build_tree(hierarchy_dict):
      """Build a tree structure from the flat hierarchy data."""
      tree = {"name": "fenic", "type": "root", "children": {}}

      # Process each element
      for i, qual_name in enumerate(hierarchy_dict['qualified_name']):
          name = hierarchy_dict['name'][i]
          elem_type = hierarchy_dict['type'][i]
          depth = hierarchy_dict['depth'][i]
          path_parts = hierarchy_dict['path_parts'][i]

          # Navigate to the correct position in the tree
          current = tree
          for j, part in enumerate(path_parts[:-1]):  # All but the last part
              if part not in current['children']:
                  current['children'][part] = {
                      "name": part,
                      "type": "unknown",  # Will be updated when we process that element
                      "children": {}
                  }
              current = current['children'][part]

          # Add the final element
          if len(path_parts) > 0:
              final_part = path_parts[-1]
              current['children'][final_part] = {
                  "name": name,
                  "type": elem_type,
                  "qualified_name": qual_name,
                  "depth": depth,
                  "children": {}
              }

      return tree

def print_tree(node, indent=0, max_depth=3):
      """Print tree structure with indentation."""
      if indent > max_depth:
          return

      if indent > 0:  # Skip root
          print("  " * (indent-1) + f"├─ [{node['type']}] {node['name']}")

      # Sort children by type then name for better readability
      children = sorted(
          node.get('children', {}).values(),
          key=lambda x: (
              0 if x['type'] == 'module' else
              1 if x['type'] == 'class' else
              2 if x['type'] == 'function' else
              3 if x['type'] == 'method' else
              4,
              x['name']
          )
      )

      for child in children[:10]:  # Limit to first 10 children
          print_tree(child, indent + 1, max_depth)

      if len(children) > 10:
          print("  " * indent + f"... and {len(children) - 10} more")

def tree_to_string(node, indent=0, max_depth=3):
      """Convert tree structure to string with indentation."""
      if indent > max_depth:
          return ""

      result = ""

      if indent > 0:  # Skip root
          result += "  " * (indent-1) + f"├─ [{node['type']}] {node['name']}\n"

      # Sort children by type then name for better readability
      children = sorted(
          node.get('children', {}).values(),
          key=lambda x: (
              0 if x['type'] == 'module' else
              1 if x['type'] == 'class' else
              2 if x['type'] == 'function' else
              3 if x['type'] == 'method' else
              4,
              x['name']
          )
      )

      for child in children[:10]:  # Limit to first 10 children
          result += tree_to_string(child, indent + 1, max_depth)

      if len(children) > 10:
          result += "  " * indent + f"... and {len(children) - 10} more\n"

      return result
  
def dataframe_to_markdown(df, max_rows: int = None) -> str:
      """
      Convert a langframe DataFrame to markdown table format.

      Args:
          df: A langframe DataFrame instance
          max_rows: Maximum number of rows to include (None for all rows)

      Returns:
          str: Markdown table representation of the DataFrame
      """
      # Collect the data as a Polars DataFrame
      polars_df = df.to_polars()

      # Limit rows if specified
      if max_rows is not None:
          polars_df = polars_df.head(max_rows)

      # Get column names
      columns = polars_df.columns

      # Start building the markdown table
      markdown = "| " + " | ".join(columns) + " |\n"
      markdown += "|" + "|".join([" --- " for _ in columns]) + "|\n"

      # Add data rows
      for row in polars_df.iter_rows():
          row_str = "| " + " | ".join([str(cell) if cell is not None else "" for cell in row]) + " |\n"
          markdown += row_str

      return markdown


#@mcp.tool()
#def search_modules(query: str) -> str:
#    """Search the Fenic project for modules that match the query. It's always best to look for keywords that are relevant to what the user is looking for, don't add descriptions or long form text, instead just 
#    look for terms that are relevant to the users query. For example, if the user is asking how to perform joins, look for the term "join".
#    """
#    session = get_session()
#    modules = session.table("modules_with_summaries").select("module", "summary")
#    resuslt = modules.filter(fc.col("summary").contains(query))
#    return dataframe_to_markdown(result)

#@mcp.tool()
#def search_functions(query: str) -> str:
#    """Search the Fenic project for functions that match the query. It's always best to look for keywords that are relevant to what the user is looking for, don't add descriptions or long form text, instead just 
#    look for terms that are relevant to the users query. For example, if the user is asking how to perform joins, look for the term "join".
#    """
#    session = get_session()
#    funcs = session.table("method_summaries").filter(~fc.col("name").starts_with("_")).select("name","qualified_name", "summary").unnest("summary").select("name", "qualified_name", "purpose", "usage_pattern")
#    result = funcs.filter(fc.col("name").contains(query) | fc.col("qualified_name").contains(query) | fc.col("purpose").contains(query) | fc.col("usage_pattern").contains(query))
#    return dataframe_to_markdown(result)

@mcp.tool()
def search(query: str, max_results: int = 30) -> str:
      """
      Search the Fenic codebase for functions, classes, methods, and other code elements.

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
          session = get_session()
          df = session.table("api_df")

          # Filter only public API elements
          df = df.filter(
              (fc.col("is_public") == True) &
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
          ).order_by([fc.col("score").desc(), fc.col("type"), fc.col("name")]).limit(max_results)

          # Collect results
          results = search_df.to_pydict()

          # Format output
          output = f"# Search Results for: `{query}`\n\n"
          output += f"Found {len(results.get('name', []))} matches\n\n"

          if not results.get('name'):
              output += "No results found. Try:\n"
              output += "- Different keywords (e.g., 'extract', 'semantic', 'DataFrame')\n"
              output += "- Regex patterns (e.g., 'join.*semantic')\n"
              return output

          # Group by type for clarity
          current_type = None
          for i in range(len(results['name'])):
              if results['type'][i] != current_type:
                  current_type = results['type'][i]
                  output += f"\n## {current_type.capitalize()}s\n"

              # Format each result concisely
              output += f"\n**`{results['name'][i]}`** - `{results['qualified_name'][i]}`\n"

              # Add first line of docstring if available
              #if results.get('docstring') and results['docstring'][i]:
               #   first_line = results['docstring'][i].strip().split('\n')[0]
                #  if len(first_line) > 100:
                 #     first_line = first_line[:97] + "..."
                 # output += f"  {first_line}\n"
              if results.get('docstring') and results['docstring'][i]:
                  output += f"  {results['docstring'][i]}\n"
          #output += f"\n---\nUse `get_details(qualified_name)` to see full documentation for any result."

          return output

      except Exception as e:
          return f"Search error: {str(e)}"

@mcp.tool()
def get_project_overview() -> str:
    """Get a high-level overview of the Fenic project. This should be the starting point for figuring out where to look next for specific questions."""
    session = get_session()
    overview = session.table("fenic_summary").select("project_summary").to_pydict()["project_summary"]
    structure = session.table("hierarchy_df").filter((fc.col("is_public") == True) & (fc.col("type") != "attribute") & (~fc.col("name").starts_with("_"))).select("qualified_name", "name", "type", "depth", "path_parts").to_pydict()
    tree = tree_to_string(build_tree(structure))
    result = f"## Fenic Project Overview\n\n{overview}\n\n## Fenic API Tree\n\n{tree}"
    return result

@mcp.tool()
def get_api_tree() -> str:
    """Get the API tree of the Fenic project."""
    session = get_session()
    structure = session.table("hierarchy_df").filter((fc.col("is_public") == True) & (fc.col("type") != "attribute") & (~fc.col("name").starts_with("_"))).select("qualified_name", "name", "type", "depth", "path_parts").to_pydict()
    tree = tree_to_string(build_tree(structure))
    result = f"## Fenic API Tree\n\n{tree}"
    return result

@mcp.tool()
def store_learning(
    question: str,
    answer: str,
    learning_type: str = "solution",  # "solution", "correction", "example"
    keywords: List[str] = None,
    related_functions: List[str] = None
) -> str:
    """
    Store a learning from a user interaction for future reference.
    
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
    session = get_session()
    # Initialize table if it doesn't exist
    initialize_learnings_table(session)
    
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
if __name__ == "__main__":
    mcp.run()