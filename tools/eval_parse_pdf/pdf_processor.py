#!/usr/bin/env python3
"""
PDF Parser Script using Fenic

This script processes PDFs using Fenic's semantic parsing capabilities.
It reads PDF metadata, parses the content using semantic.parse_pdf,
saves results to a Fenic table, and writes individual markdown files.
"""

import argparse
import importlib.util
import os
from datetime import datetime

from fenic import col, semantic
from fenic.api.session.config import (
    GoogleDeveloperLanguageModel,
    OpenAILanguageModel,
    OpenRouterLanguageModel,
    SemanticConfig,
    SessionConfig,
)
from fenic.api.session.session import Session


def process_pdfs(input_glob_pattern, test_id, output_dir, app_name, model_name, parse_engine="native", session=None):
    """
    Process PDFs using Fenic semantic parsing.
    
    Args:
        input_glob_pattern (str): Glob pattern for input PDF files
        test_id (str): Test identifier for naming outputs
        output_dir (str): Directory to save output files
        app_name (str): Application name for Fenic session
        model_name (str): Name of the language model to use for parsing
        model_alias (str): Model alias for semantic parsing
        parse_engine (str): Parse engine to use for parsing PDFs (for OpenRouter models only). Options are: mistral-ocr, pdf-text, native
    """
    print(f"Starting PDF processing with test_id: {test_id}")
    print(f"Input pattern: {input_glob_pattern}")
    print(f"Output directory: {output_dir}")
    print(f"App name: {app_name}")
    print(f"Model name: {model_name}")
    print(f"Parse engine: {parse_engine}")
    now = datetime.now()
    close_session = True
    
    # Initialize Fenic session with semantic configuration
    model_components = model_name.split("/")
    model_alias = model_name
    if len(model_components) == 3:
        model_alias = f"{model_components[0]}/{model_components[1]}"
    if session is None:
        # Configure language models
        language_models = get_model_configs()

        # Configure OpenRouter models separately
        # The model will be in the format of {provider}/{model}
        # You can pass the parse engine as an optional argument to the function (default is native), or as part of the model_name string: {provider}/{model}/{parse_engine}

        if len(model_components) == 2:
            language_models[model_name] = build_openrouter_model_config(model_name, parse_engine)
        elif len(model_components) == 3:
            engine_component = model_components[2]
            language_models[model_alias] = build_openrouter_model_config(model_alias, engine_component)
        elif model_name not in language_models:
            available_models = list(language_models.keys())
            raise ValueError(f"Model '{model_name}' not supported. Available models: {available_models}")

        # Initialize Fenic session with semantic configuration
        semantic_config = SemanticConfig(
            language_models=language_models,
            default_language_model="gpt-5-nano"  # Default model
        )
        
        config = SessionConfig(
            app_name=app_name,
            semantic=semantic_config
        )
        session = Session.get_or_create(config)
    else:
        close_session = False

    # Step 0: Create table name and check if table already exists.
    table_name = f"parsed_output_test_{test_id}_{model_name}"
    table_name = table_name.replace("/", "__")
    table_name = table_name.replace(".", "_")
    parsed_list = None
    try:
        parsed_df = session.table(table_name)
        parsed_list = parsed_df.collect("pylist").data
    except ValueError:
        print(f"table {table_name} does not exist yet.")

    if parsed_list is None:
        # Step 1: Read PDF metadata
        print("Reading PDF metadata...")
        metadata_df = session.read.pdf_metadata(input_glob_pattern, recursive=True).cache()
        print(f"Found {metadata_df.count()} PDF files")
        
        if metadata_df.count() == 0:
            print("No PDF files found matching the pattern. Exiting.")
            return
        
        # Step 2: Parse PDFs using semantic.parse_pdf and cache the result
        print(f"Parsing PDF content using model: {model_name}")
        df = metadata_df.select(
            col("title"),
            col("file_path"),
            semantic.parse_pdf(col("file_path"), model_alias=model_alias).alias("parsed_output")
        ).cache()
        result = df.collect("pylist")
        parsed_list = result.data
        print(f"Parsed {len(parsed_list)} PDFs successfully")
        cost_result_df = session.create_dataframe(
            {
                "test_id": [test_id], 
                "model_name": [model_name],
                "execution_id": [result.metrics.execution_id],
            })
        
        # Step 3: Write to Fenic table using test_id
        print(f"Saving to Fenic table: {table_name}")
        df.write.save_as_table(table_name, mode="overwrite")
        cost_result_df.write.save_as_table("eval_execution_ids", mode="append")


    # Step 4: Create output directory structure
    test_output_dir = output_dir
    os.makedirs(test_output_dir, exist_ok=True)
    print(f"Created output directory: {test_output_dir}")

    # Step 5: Write individual markdown files
    print("Writing individual markdown files...")
    files_written = 0


    for row in parsed_list:
        title = row.get('title', 'untitled')
        parsed_output = row.get('parsed_output', '')
        
        # Create markdown filename directly from title
        safe_title = str(title) if title else f"untitled_{files_written}"
        md_filename = f"{safe_title}.md"
        output_path = os.path.join(test_output_dir, md_filename)
        
        # Write parsed output to markdown file (overwrite if exists)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(str(parsed_output))
            files_written += 1
            print(f"  Written: {md_filename}")
        except Exception as e:
            print(f"  Error writing {md_filename}: {e}")
    
    print("\nProcessing complete!")
    print(f"- Processed {len(parsed_list)} PDFs")
    print(f"- Saved to table: {table_name}")
    print(f"- Written {files_written} markdown files to: {test_output_dir}")
    print(f"- Time taken: {datetime.now() - now}")

    if close_session:
        session.stop()

def build_openrouter_model_config(model_name, parse_engine="native"):
    return OpenRouterLanguageModel(
        model_name=model_name,
        profiles={"default": OpenRouterLanguageModel.Profile(
            parsing_engine=parse_engine
        )}
    )

def get_model_configs():
    model_configs = {
        # Google Developer models

        # OpenAI models
        "o3": OpenAILanguageModel(
            model_name="o3",
            rpm=100,
            tpm=200000,
            profiles={"minimal": OpenAILanguageModel.Profile(
                reasoning_effort="low"
            )},
            default_profile="minimal"
        ),
        "o4-mini": OpenAILanguageModel(
            model_name="o4-mini",
            rpm=100,
            tpm=200000,
            profiles={"minimal": OpenAILanguageModel.Profile(
                reasoning_effort="low"
            )},
            default_profile="minimal"
        ),
        "gpt-4o": OpenAILanguageModel(
            model_name="gpt-4o",
            rpm=100,
            tpm=200000,
        ),
        "gpt-4o-mini": OpenAILanguageModel(
            model_name="gpt-4o-mini",
            rpm=100,
            tpm=200000,
        ),
        "gpt-5-nano": OpenAILanguageModel(
            model_name="gpt-5-nano",
            rpm=100,
            tpm=200000,
            profiles={"minimal": OpenAILanguageModel.Profile(
                reasoning_effort="minimal",
                verbosity="low",
            )},
            default_profile="minimal"
        ),
        "gpt-5-mini": OpenAILanguageModel(
            model_name="gpt-5-mini",
            rpm=100,
            tpm=200000,
            profiles={"minimal": OpenAILanguageModel.Profile(
                reasoning_effort="minimal",
                verbosity="low",
            )},
            default_profile="minimal"
        ),
        "gpt-5": OpenAILanguageModel(
            model_name="gpt-5",
            rpm=100,
            tpm=200000,
            profiles={"minimal": OpenAILanguageModel.Profile(
                reasoning_effort="minimal",
                verbosity="low",
            )},
            default_profile="minimal"
        ),
    }
    if importlib.util.find_spec("google.genai") is not None:
        model_configs["gemini-2.5-flash"] = GoogleDeveloperLanguageModel(
            model_name="gemini-2.5-flash",
            rpm=100,
            tpm=200000,
            profiles={"disabled_thinking": GoogleDeveloperLanguageModel.Profile(
                thinking_token_budget=0
            )},
            default_profile="disabled_thinking"
        )
        model_configs["gemini-2.5-pro"] = GoogleDeveloperLanguageModel(
            model_name="gemini-2.5-pro",
            rpm=100,
            tpm=200000,
            profiles={"minimal_thinking": GoogleDeveloperLanguageModel.Profile(
                thinking_token_budget=128
            )},
            default_profile="minimal_thinking"
        )
        model_configs["gemini-2.0-flash"] = GoogleDeveloperLanguageModel(
            model_name="gemini-2.0-flash",
            rpm=100,
            tpm=200000,
            profiles={"disabled_thinking": GoogleDeveloperLanguageModel.Profile(
                thinking_token_budget=0
            )},
            default_profile="disabled_thinking"
        )
        model_configs["gemini-2.0-flash-lite"] = GoogleDeveloperLanguageModel(
            model_name="gemini-2.0-flash-lite",
            rpm=100,
            tpm=200000,
            profiles={"disabled_thinking": GoogleDeveloperLanguageModel.Profile(
                thinking_token_budget=0
            )},
            default_profile="disabled_thinking"
        )
        model_configs["gemini-2.5-flash-lite"] = GoogleDeveloperLanguageModel(
            model_name="gemini-2.5-flash-lite",
            rpm=100,
            tpm=200000,
            profiles={"disabled_thinking": GoogleDeveloperLanguageModel.Profile(
                thinking_token_budget=0
            )},
            default_profile="disabled_thinking"
        )
        model_configs["gemini-3-pro-preview"] = GoogleDeveloperLanguageModel(
            model_name="gemini-3-pro-preview",
            rpm=100,
            tpm=200000,
            profiles={"disabled_thinking": GoogleDeveloperLanguageModel.Profile(
                thinking_level="low",
                media_resolution="low"
            )},
            default_profile="disabled_thinking"
        )
        model_configs["gemini-3-pro-preview-high-res"] = GoogleDeveloperLanguageModel(
            model_name="gemini-3-pro-preview",
            rpm=100,
            tpm=200000,
            profiles={"disabled_thinking": GoogleDeveloperLanguageModel.Profile(
                thinking_level="low",
                media_resolution="high"
            )},
            default_profile="disabled_thinking"
        )
    return model_configs

def main():
    """Main function to handle command line arguments and orchestrate processing."""
    parser = argparse.ArgumentParser(
        description="Process PDFs using Fenic semantic parsing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval_parse_pdf_processor.py --input-glob-pattern "/data/pdfs/**/*.pdf" --app-name my_pdf_app --model-name gemini-2.0-flash --test-id experiment_1 --output-dir /output/results
  
  python eval_parse_pdf_processor.py -i "docs/*.pdf" -a pdf_processor -m o3-mini -t test_batch_2 -o ./parsed_outputs --model-alias custom_parser
  
  python eval_parse_pdf_processor.py --input-glob-pattern "/research/papers/**/*.pdf" --app-name research_app --model-name gemini-2.5-pro --test-id paper_analysis --output-dir /results/analysis
        """
    )
    
    parser.add_argument(
        '--input-glob-pattern', '-i',
        required=True,
        help='Glob pattern for input PDF files (e.g., "/data/pdfs/**/*.pdf")'
    )
    
    parser.add_argument(
        '--app-name', '-a',
        required=True,
        help='Application name for the Fenic session'
    )
    
    parser.add_argument(
        '--model-name', '-m',
        required=True,
        help='Name of the language model to use for parsing (e.g., gemini-2.0-flash, o3-mini)'
    )
    
    parser.add_argument(
        '--test-id', '-t',
        required=True,
        help='Test identifier for naming outputs and organizing results'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        required=True,
        help='Output directory path where results will be saved'
    )
    
    parser.add_argument(
        '--parse-engine', '-e',
        default='native',
        choices=['mistral-ocr', 'pdf-text', 'native'],
        help='Parse engine to use for parsing PDFs (for OpenRouter models only). Options are: mistral-ocr, pdf-text, native'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.test_id.strip():
        print("Error: test_id cannot be empty")
        return 1
    
    process_pdfs(
        input_glob_pattern=args.input_glob_pattern,
        test_id=args.test_id,
        output_dir=args.output_dir,
        app_name=args.app_name,
        model_name=args.model_name,
        parse_engine=args.parse_engine
    )
    return 0


if __name__ == "__main__":
    main()
