#!/usr/bin/env python3
"""
PDF Parsing Evaluation Test Harness

This script orchestrates the evaluation of PDF parsing across multiple models,
running the parsing and grading processes, and collecting results from Fenic.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import polars as pl
from grade_md import grade_pdfs

# Import the processing functions directly
from pdf_processor import build_openrouter_model_config, get_model_configs, process_pdfs

import fenic as fc
from fenic.api.session.config import (
    SemanticConfig,
    SessionConfig,
)
from fenic.api.session.session import Session


class PDFEvalTestHarness:
    """Test harness for evaluating PDF parsing across multiple models."""
    
    def __init__(
        self,
        app_name: str,
        models: List[str],
        pdf_glob: str,
        markdown_output_path: str,
        test_id: str,
        verbose: bool = False
    ):
        """
        Initialize the test harness.
        
        Args:
            app_name: Name of the application (e.g., "evaluate_pdf_parsing")
            models: List of model names to evaluate
            pdf_glob: Glob pattern for PDF files
            markdown_output_path: Base path for markdown output
            test_id: Test ID for this evaluation run
            verbose: Enable verbose output
        """
        self.app_name = app_name
        self.models = models
        self.pdf_glob = pdf_glob
        self.markdown_output_path = Path(markdown_output_path)
        self.test_id = test_id
        self.verbose = verbose
        self.session = None
        
        # Initialize Fenic session
        try:
            self.configure_fenic_session(self.app_name, self.models)
        except:
            print("Error: Could not initialize Fenic session.")
            raise
    
    def run_model_evaluation(self, model: str) -> dict:
        """
        Run complete evaluation for a single model.
        
        Args:
            model: Model name
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'#'*70}")
        print(f"# Evaluating Model: {model}")
        print(f"# Test ID: {self.test_id}")
        print(f"{'#'*70}")
        
        result = {
            "model": model,
            "test_id": self.test_id,
            "pdf_processing": False,
            "grading": False,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Process PDFs
            print(f"\nProcessing PDFs with model {model}...")
            model_components = model.split("/")
            md_output_dir = self.markdown_output_path / self.test_id / "__".join(model_components)
            process_pdfs(
                input_glob_pattern=self.pdf_glob,
                test_id=self.test_id,
                output_dir=str(md_output_dir),
                app_name=self.app_name,
                model_name=model,
                session=self.session
            )
            result["pdf_processing"] = True
            print(f"✓ PDF processing completed for {model}")
            
            # Grade the markdown output
            print(f"\nGrading markdown for test {self.test_id}...")
            grade_pdfs(
                pdf_dir=self.pdf_glob,
                md_dir=str(md_output_dir),
                test_id=self.test_id,
                app_name=self.app_name,
                model_name=model,
                session=self.session
            )
            result["grading"] = True
            print(f"✓ Grading completed for {self.test_id}")
        
        except KeyboardInterrupt:
            print(f"Stopping evaluation during {model}: KeyboardInterrupt")
            if self.session:
                self.session.stop()
            return result
        except Exception as e:
            print(f"✗ Error evaluating model {model}: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
        return result
    
    def retrieve_results_from_fenic(self) -> Optional[fc.DataFrame]:
        """
        Retrieve evaluation results from Fenic table.

        Joins the metrics from the pdf_parse for that model with the eval_results
        
        Returns:
            DataFrame with results or None if unable to retrieve
        """
        if not self.session:
            print("Fenic session not available. Skipping results retrieval.")
            return None
        
        try:
            # Read the eval_results table
            eval_results = self.session.table("eval_results")
            
            # Filter for exact test_id match
            filtered_results = eval_results.filter(
                fc.col("test_id") == self.test_id
            ).select(
                "model_name",
                "test_id",
                "overall_score",
                "structure_f1",
                "text_score",
                "total_pages"
            )
            # join with eval_query_ids on the model_name and test_id
            eval_execution_ids = self.session.table("eval_execution_ids")
            eval_results_with_query_ids = filtered_results.join(eval_execution_ids, on=["model_name", "test_id"], how="left")

            # join with metrics using query_id
            metrics_df = self.session.table("fenic_system.query_metrics").select(
                "execution_id",
                "total_lm_cost",
                "total_lm_uncached_input_tokens",
                "total_lm_cached_input_tokens",
                "total_lm_output_tokens",
                "total_lm_requests",
                "execution_time_ms"
            )
            eval_results_complete = eval_results_with_query_ids.join(metrics_df, on="execution_id", how="left")
            return eval_results_complete.select(
                "model_name",
                "overall_score", 
                "structure_f1",
                "text_score",
                fc.col("total_lm_cost").alias("lm_cost"),
                fc.col("execution_time_ms").alias("time_ms"),
                fc.col("total_lm_uncached_input_tokens").alias("uncached_input_tokens"),
                fc.col("total_lm_cached_input_tokens").alias("cached_input_tokens"),
                fc.col("total_lm_output_tokens").alias("output_tokens"),
                "total_pages"
            )
            
        except Exception as e:
            print(f"Error retrieving results from Fenic: {e}")
            return None

    def clean_eval_results(self):
        """
        Clean the eval_results table for this test_id
        """
        if self.session.catalog.does_table_exist("eval_results"):
            self.session.table("eval_results").filter(
                fc.col("test_id") != self.test_id
            ).write.save_as_table("eval_results", mode="overwrite")
        if self.session.catalog.does_table_exist("eval_execution_ids"):
            self.session.table("eval_execution_ids").filter(
                fc.col("test_id") != self.test_id
            ).write.save_as_table("eval_execution_ids", mode="overwrite")


    def run(self) -> List[dict]:
        """
        Run the complete test harness for all models.
        
        Returns:
            List of result dictionaries for each model
        """
        print("\nStarting PDF Evaluation Test Harness")
        print(f"Test ID: {self.test_id}")
        print(f"Models to evaluate: {', '.join(self.models)}")
        print(f"PDF glob: {self.pdf_glob}")
        print(f"Output path: {self.markdown_output_path}")

        # Ensure output directory exists
        self.markdown_output_path.mkdir(parents=True, exist_ok=True)

        self.clean_eval_results()

        # Run evaluation for each model
        all_results = []
        for model in self.models:
            result = self.run_model_evaluation(model)
            all_results.append(result)
        
        # Display processing summary
        print(f"\n{'='*70}")
        print("PROCESSING SUMMARY")
        print(f"{'='*70}")
        
        for result in all_results:
            status = "✓" if result["pdf_processing"] and result["grading"] else "✗"
            print(f"{status} {result['model']}: "
                  f"PDF={'✓' if result['pdf_processing'] else '✗'} "
                  f"Grade={'✓' if result['grading'] else '✗'}")
        
        # Retrieve and display results from Fenic
        results_df = self.retrieve_results_from_fenic()
        if results_df:
            print(f"\n{'='*70}")
            print(f"EVALUATION RESULTS FOR TEST_ID: {self.test_id}")
            print(f"{'='*70}")
            pl.Config.set_fmt_str_lengths(200)
            pl.Config.set_tbl_width_chars(200)
            pl.Config.set_tbl_cols(-1)  
            print(results_df.to_polars())
            breakpoint()
        else:
            print("\nNo results to display from Fenic.")
        
        # Save processing summary
        summary_file = self.markdown_output_path / f"{self.test_id}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nProcessing summary saved to: {summary_file}")
        
        return all_results

    

    def configure_fenic_session(self, app_name: str, models: List[str]):
        """
        Configure the Fenic session.
        """
        # Configure language models
        language_models = get_model_configs()

        for model_name in models:
            # Configure OpenRouter models separately
            # The model will be in the format of {provider}/{model}
            # You can pass the parse engine as an optional argument to the function (default is native), or as part of the model_name string: {provider}/{model}/{parse_engine}
            model_components = model_name.split("/")
            model_alias = model_name
            if len(model_components) == 2:
                language_models[model_name] = build_openrouter_model_config(model_name)
            elif len(model_components) == 3:
                model_alias = f"{model_components[0]}/{model_components[1]}"
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
        self.session = Session.get_or_create(config)

    def close(self):
        if self.session:
            self.session.stop()


def main():
    """Main entry point for the test harness."""
    parser = argparse.ArgumentParser(
        description="PDF Parsing Evaluation Test Harness"
    )
    
    parser.add_argument(
        "-a", "--app-name",
        required=True,
        help="Application name (e.g., 'evaluate_pdf_parsing')"
    )
    
    parser.add_argument(
        "-m", "--models",
        nargs="+",
        required=True,
        help="List of model names to evaluate"
    )
    
    parser.add_argument(
        "-i", "--input-pdfs",
        required=True,
        help="Glob pattern for input PDF files"
    )
    
    parser.add_argument(
        "-o", "--output-path",
        required=True,
        help="Base path for markdown output"
    )
    
    parser.add_argument(
        "-t", "--test-id",
        required=True,
        help="Test ID for this evaluation run"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output with stack traces"
    )
    
    args = parser.parse_args()
    
    # Create and run the test harness
    harness = PDFEvalTestHarness(
        app_name=args.app_name,
        models=args.models,
        pdf_glob=args.input_pdfs,
        markdown_output_path=args.output_path,
        test_id=args.test_id,
        verbose=args.verbose
    )
    
    results = harness.run()
    harness.close()
    
    # Exit with error if any evaluations failed
    if any(not (r["pdf_processing"] and r["grading"]) for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()