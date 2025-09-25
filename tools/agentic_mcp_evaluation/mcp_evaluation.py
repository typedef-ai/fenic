#!/usr/bin/env python3
"""MCP Server Evaluation Script

This script adapts the tool evaluation framework to test MCP servers created with Fenic.
It creates an MCP server without running it, extracts tool schemas, and evaluates the tools
using an AI agent that can call the tools programmatically.

Usage:
    python tools/mcp_evaluation.py --eval-file evaluation.xml --mcp-config mcp_demo_setup.py
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
import traceback
import xml.etree.ElementTree as ET  #nosec B405: file is local and trusted within repo
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult
from mcp_to_evaluate import setup_mcp_for_evaluation
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

import fenic as fc
from fenic.api.session.config import OpenRouterLanguageModel
from fenic.core.mcp._server import MCPResultSet
from fenic.core.metrics import LMMetrics
from fenic.core.types.datatypes import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

logger = logging.getLogger(__name__)

# Embedded evaluator prompts
AGENT_PROMPT = """You are an expert data analyst with access to a toolset that allows you to query and analyze data.

When you are first analyzing the data, you SHOULD:
- Use the `Schema` tool to understand the structure of the data
- Use the `Profile` tool to understand the contents/distribution of the data

When given a task, you MUST:
1. Use the available tools to complete the task
2. Provide summary of each step in your approach, wrapped in <summary> tags
3. Provide feedback on the tools provided, wrapped in <feedback> tags
4. Provide your final response, wrapped in <response> tags

IMPORTANT:
- Be conservative with token usage, tools are sampling data that contains long text fields.

Summary Requirements:
- In your <summary> tags, you must explain:
  - The steps you took to complete the task
  - Which tools you used, in what order, and why
  - The inputs you provided to each tool
  - The outputs you received from each tool
  - A summary for how you arrived at the response

Feedback Requirements:
- In your <feedback> tags, provide constructive feedback on the tools:
  - Comment on tool names: Are they clear and descriptive?
  - Comment on input parameters: Are they well-documented? Are required vs optional parameters clear?
  - Comment on descriptions: Do they accurately describe what the tool does?
  - Comment on any errors encountered during tool usage: Did the tool fail to execute? Did the tool return too many tokens?
  - Identify specific areas for improvement and explain WHY they would help
  - Be specific and actionable in your suggestions
  - The feedback should be from the perspective of an AI agent that is using the tools, not a human user.
  
Response Requirements:
- Your response should be concise and directly address what was asked
- Always wrap your final response in <response> tags
- If you cannot solve the task return <response>NOT_FOUND</response>
- For numeric responses, provide just the number
- For IDs, provide just the ID
- For names or text, provide the exact text requested
- Your response should go last"""

SCORING_PROMPT = """You are an expert evaluator. Score the assistant's self-reported <summary> and <feedback> for quality of tool usage and guidance.


Rubric:
- clarity: clear, structured, easy to follow
- actionability: specific, concrete next steps for improving tools/usage
- correctness: technically sound about tools, parameters, constraints and limits
- coverage: addresses key aspects (ordering: Schema -> Profile, params, pitfalls, paging)
- safety: identifies risks (oversized results, destructive ops) and mitigations
"""


class ScoreCriteria(BaseModel):
    clarity: float = Field(description="Clarity of explanation and structure (0-10)", ge=0, le=10)
    actionability: float = Field(description="Actionable next steps (0-10)", ge=0, le=10)
    correctness: float = Field(description="Technical correctness about tools/limits (0-10)", ge=0, le=10)
    coverage: float = Field(description="Covers key aspects: ordering, params, pitfalls (0-10)", ge=0, le=10)
    safety: float = Field(description="Addresses risks like oversized reads (0-10)", ge=0, le=10)

def __str__(self) -> str:
        return f"Clarity: {self.clarity}\nActionability: {self.actionability}\nCorrectness: {self.correctness}\nCoverage: {self.coverage}\nSafety: {self.safety}"

class FeedbackScore(BaseModel):
    overall: float = Field(description="Overall score 0-10", ge=0, le=10)
    criteria: ScoreCriteria = Field(description="Underlying scores for each criteria that inform the overall score")
    justification: str = Field(description="Justification for the overall score and underlying scores")

    def __str__(self) -> str:
        return f"Overall score: {self.overall}\nCriteria: ({self.criteria})\nJustification: {self.justification}"


class AgentLoopResult(BaseModel):
    response: str
    tool_metrics: Dict[str, Any]
    agent_usage: LMMetrics

class EvaluateSingleTaskResult(BaseModel):
    prompt: str
    response: str
    total_duration: float
    tool_metrics: Dict[str, Any]
    num_tool_calls: int
    agent_usage: LMMetrics
    summary: str
    feedback: str

class ScoredTaskResult(BaseModel):
    task_result: EvaluateSingleTaskResult
    score: FeedbackScore


class MCPEvaluator:
    """Evaluates MCP servers by creating them and testing their tools."""

    def __init__(self, agent_model: str | None = None, evaluation_model: str | None = None, session: fc.Session | None = None):
        # OpenRouter (OpenAI-compatible) client
        # Requires OPENROUTER_API_KEY in environment
        self.client = AsyncOpenAI(
            base_url=os.environ.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        self.agent_model = agent_model
        self.evaluation_model = evaluation_model
        self.session = session
        self.server = None
        self.tools: List[Dict[str, Any]] = []

    async def setup_mcp_server(self) -> None:
        """Set up the MCP server from a configuration file."""
        logger.info("🔧 Setting up MCP server")

        # Extract tool schemas
        fc.logging.configure_logging()
        self.server = setup_mcp_for_evaluation(self.session)
        self.tools = await self._extract_tool_schemas()
        logger.info(f"✅ Loaded {len(self.tools)} tools from MCP server")

    async def _extract_tool_schemas(self) -> List[Dict[str, Any]]:
        """Extract JSON schemas for all tools in the MCP server."""
        tools = []
        # Get tools from the FastMCP instance
        for tool_name, tool_info in (await self.server.mcp.get_tools()).items():
            logger.info(f"🔍 Tool: {tool_name} - {tool_info.description}")
            # OpenAI tool schema
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_info.description,
                        "parameters": tool_info.parameters
                    },
                }
            )

        return tools


    async def _call_mcp_tool(self, tool_name: str, **kwargs) -> MCPResultSet:
        """Call a tool on the MCP server."""
        try:
            # Get the tool function from the FastMCP server
            tool_obj = await self.server.mcp.get_tool(tool_name)
            if not tool_obj:
                return f"Error: Tool '{tool_name}' not found"
            logger.info(f"🔍 Calling tool {tool_name} with kwargs: {kwargs}")
            # Call the tool function using the run method
            result: ToolResult = await tool_obj.run(kwargs)
            parsed_result = MCPResultSet(**result.structured_content)
            logger.debug(f"🔍 Tool {tool_name} returned: {parsed_result}")
            return parsed_result

        except ToolError:
            # Deeply format chained exceptions and extract domain-specific context
            exc_type, exc_value, exc_traceback = sys.exc_info()
            formatted_error_list = traceback.format_exception(exc_type, exc_value, exc_traceback)
            logger.error(f"Tool {tool_name} failed: {formatted_error_list}")
            return f"Tool {tool_name} failed: {formatted_error_list}"


    async def agent_loop(self, prompt: str) -> Optional[AgentLoopResult]:
        """Simplified agent loop using OpenAI Chat Completions with tool calling."""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": AGENT_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Track tool calls with timing
        tool_metrics: Dict[str, Dict[str, Any]] = {}
        agent_usage = LMMetrics()
        while True:
            resp = await self.client.chat.completions.create(
                model=self.agent_model,
                messages=messages,
                tools=self.tools if self.tools else None,
                reasoning_effort="low",
                # tool_choice="auto",
                extra_body={
                    # Preserve usage metrics
                    "usage": {
                        "include": True
                    },  
                    # Ensure providers validate tool params
                    "provider" : {
                            "require_parameters" : True
                        },
                },
            )
            usage = resp.usage
            model_extra = usage.model_extra
            cost_value = model_extra.get("cost")
            response_metrics = LMMetrics(
                num_uncached_input_tokens=usage.prompt_tokens - usage.prompt_tokens_details.cached_tokens,
                num_cached_input_tokens=usage.prompt_tokens_details.cached_tokens,
                num_output_tokens=usage.completion_tokens,
                cost=cost_value,
                num_requests=1
            )
            agent_usage += response_metrics
            choice = resp.choices[0]
            msg = choice.message
            messages.append(msg.to_dict())
            # If there are tool calls, execute them and loop
            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        args = {}

                    start_ts = time.time()
                    try:
                        tool_result = await self._call_mcp_tool(tool_name, **args)
                        if isinstance(tool_result, MCPResultSet):
                            result_text = json.dumps(tool_result.model_dump(), indent=2)
                        else:
                            result_text = str(tool_result)
                    except Exception as e:
                        result_text = f"Error executing tool {tool_name}: {e}\n{traceback.format_exc()}"
                    duration = time.time() - start_ts

                    # Update metrics
                    tool_metrics.setdefault(tool_name, {"count": 0, "durations": []})
                    tool_metrics[tool_name]["count"] += 1
                    tool_metrics[tool_name]["durations"].append(duration)

                    # Append tool result message
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tool_name,
                            "content": result_text,
                        }
                    )
                # Continue the loop for the model to process tool results
                continue

            # No tool calls; return final content
            final_text = msg.content or ""
            return AgentLoopResult(response=final_text, tool_metrics=tool_metrics, agent_usage=agent_usage)


def parse_evaluation_file(file_path: Path) -> List[Dict[str, Any]]:
    """Parse XML evaluation file and return list of evaluation tasks (prompt only)."""
    try:
        # Fallback to stdlib XML parsing; file is local and trusted within repo
        tree = ET.parse(file_path) #nosec B314: file is local and trusted within repo
        root = tree.getroot()
        evaluations = []

        # Check for task elements
        tasks = root.findall(".//task")
        for task in tasks:
            prompt_elem = task.find("prompt")
            response_elem = task.find("response")  # deprecated; ignored for scoring

            if prompt_elem is not None:
                eval_dict = {
                    "prompt": (prompt_elem.text or "").strip(),
                    # kept for backward compat; not used in scoring-only mode
                    "response": (response_elem.text or "").strip() if response_elem is not None else "",
                }
                evaluations.append(eval_dict)

        return evaluations
    except Exception as e:
        logger.error(f"Error parsing evaluation file {file_path}: {e}")
        return []



async def evaluate_single_task(
    evaluator: MCPEvaluator, task: Dict[str, Any], task_index: int
) -> EvaluateSingleTaskResult:
    """Evaluate a single task with the MCP evaluator."""
    start_time = time.time()

    # Run the task
    logger.info(f"Task {task_index + 1}: Running task with prompt: {task['prompt']}")
    result = await evaluator.agent_loop(task["prompt"])

    # Extract all tagged content
    def _extract_xml_content(text, tag):
        pattern = rf"<{tag}>(.*?)</{tag}>"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[-1].strip() if matches else None

    response_text, summary, feedback = (
        _extract_xml_content(result.response, tag)
        for tag in ["response", "summary", "feedback"]
    )
    duration_seconds = time.time() - start_time

    return EvaluateSingleTaskResult(
        prompt=task["prompt"],
        response=response_text,
        total_duration=duration_seconds,
        tool_metrics=result.tool_metrics,
        num_tool_calls=sum(len(metrics["durations"]) for metrics in result.tool_metrics.values()),
        agent_usage=result.agent_usage,
        summary=summary or "",
        feedback=feedback or "",
    )


# Jinja2 template for the full report
# Note: Fenic's Jinja doesn't support filters, so numbers are pre-formatted
REPORT_JINJA_TEMPLATE = """# MCP Server Evaluation Report

**Agent Model**: {{ agent_model }}  
**Evaluation Model**: {{ evaluation_model }}

## Summary

- **Average Task Duration**: {{ average_duration_s }}s
- **Average Tool Calls per Task**: {{ average_tool_calls }}
- **Total Tool Calls**: {{ total_tool_calls }}
- **Average Score**: {{ average_score }}/10
 - **Total Tasks**: {{ total_tasks }}

### Total Agent Usage
- Uncached Input Tokens: {{ total_uncached_input_tokens }}
- Cached Input Tokens: {{ total_cached_input_tokens }}
- Output Tokens: {{ total_output_tokens }}
- Requests: {{ total_requests }}
- Cost: ${{ total_cost }}

---

{% for task in tasks %}
### Task {{ loop.index }}

**Prompt**: {{ task.prompt }}
**Response**: `{{ task.response }}`
**Duration**: {{ task.duration }}s
**Tool Calls**: {{ task.tool_calls_json }}
**Score**: {{ task.score }}/10
**Criteria**:
- Clarity: {{ task.clarity }}/10
- Actionability: {{ task.actionability }}/10
- Correctness: {{ task.correctness }}/10
- Coverage: {{ task.coverage }}/10
- Safety: {{ task.safety }}/10

**Agent Usage**
- Summary: {{ task.agent_usage_summary }}
- Uncached Input Tokens: {{ task.uncached_input_tokens }}
- Cached Input Tokens: {{ task.cached_input_tokens }}
- Output Tokens: {{ task.output_tokens }}
- Requests: {{ task.requests }}
- Cost: ${{ task.cost }}

**Summary**
{{ task.summary }}

**Feedback**
{{ task.feedback }}

---
{% endfor %}

## Cross-Task Synthesis

### Salient Points
{{ salient_points }}

### Action Items (Grouped by Priority)
{{ action_items }}
"""


async def run_mcp_evaluation(eval_path: str, agent_model: str, evaluation_model: str, session: fc.Session) -> str:
    """Run evaluation with MCP server tools."""
    logger.info("🚀 Starting MCP Server Evaluation")

    eval_file = Path(eval_path)

    # Parse evaluation tasks
    tasks = parse_evaluation_file(eval_file)
    logger.info(f"📋 Loaded {len(tasks)} evaluation tasks")

    # Set up MCP evaluator
    evaluator = MCPEvaluator(agent_model, evaluation_model, session)
    await evaluator.setup_mcp_server()

    # Build a single Fenic pipeline for: agent execution → scoring → aggregation → templating
    
    # UDFs: format helpers
    format_2dp = fc.udf(lambda x: "{:.2f}".format(float(x)) if x is not None else "0.00", return_type=StringType)
    format_1dp = fc.udf(lambda x: "{:.1f}".format(float(x)) if x is not None else "0.0", return_type=StringType)

    # Async UDF: run the agent for a single prompt (IO-bound, concurrent)
    @fc.async_udf(
        return_type=StructType(
            [
                StructField("response", StringType),
                StructField("summary", StringType),
                StructField("feedback", StringType),
                StructField("total_duration", FloatType),
                StructField("tool_metrics_json", StringType),
                StructField("num_tool_calls", IntegerType),
                # Agent usage metrics
                StructField("uncached_input_tokens", IntegerType),
                StructField("cached_input_tokens", IntegerType),
                StructField("output_tokens", IntegerType),
                StructField("requests", IntegerType),
                StructField("cost", FloatType),
                StructField("agent_usage_summary", StringType),
            ]
        ),
        max_concurrency=32,
        timeout_seconds=300,
        num_retries=0,
    )
    async def run_agent(prompt: str) -> Dict[str, Any]:
        start_ts = time.time()
        result = await evaluator.agent_loop(prompt)
        usage = result.agent_usage
        logger.info(f"""\
        🔍 Task Agent usage:
            Uncached Input Tokens: {usage.num_uncached_input_tokens}
            Cached Input Tokens: {usage.num_cached_input_tokens}
            Output Tokens: {usage.num_output_tokens}
            Requests: {usage.num_requests}
            Cost: ${usage.cost:.2f}
        """)
        # Extract tagged blocks
        def _extract_xml_content(text: str, tag: str) -> str:
            try:
                matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text or "", re.DOTALL)
                return matches[-1].strip() if matches else ""
            except Exception:
                return ""

        summary = _extract_xml_content(result.response, "summary")
        feedback = _extract_xml_content(result.response, "feedback")
        duration = time.time() - start_ts
        tool_calls = sum(len(m.get("durations", [])) for m in result.tool_metrics.values())
        # Agent usage metrics
        usage = result.agent_usage
        try:
            summary_text = usage.get_summary()  # type: ignore[attr-defined]
        except Exception:
            summary_text = str(usage)
        return {
            "response": _extract_xml_content(result.response, "response"),
            "summary": summary,
            "feedback": feedback,
            "total_duration": float(duration),
            "tool_metrics_json": json.dumps(result.tool_metrics, indent=2),
            "num_tool_calls": tool_calls,
            "uncached_input_tokens": usage.num_uncached_input_tokens,
            "cached_input_tokens": usage.num_cached_input_tokens,
            "output_tokens": usage.num_output_tokens,
            "requests": usage.num_requests,
            "cost": usage.cost,
            "agent_usage_summary": summary_text,
        }

    # Create task DataFrame
    tasks_df = session.create_dataframe({
        "prompt": [t["prompt"] for t in tasks],
    })

    # Full pipeline
    pipeline_df = (
        tasks_df
        .select(
            fc.col("prompt"),
            run_agent(fc.col("prompt")).alias("agent"),
        )
        .select(
            fc.col("prompt"),
            fc.col("agent")["response"].alias("response"),
            fc.col("agent")["summary"].alias("summary"),
            fc.col("agent")["feedback"].alias("feedback"),
            fc.col("agent")["total_duration"].alias("duration_raw"),
            fc.col("agent")["tool_metrics_json"].alias("tool_calls_json"),
            fc.col("agent")["num_tool_calls"].alias("num_tool_calls"),
            # Per-task usage metrics
            fc.col("agent")["uncached_input_tokens"].alias("uncached_input_tokens"),
            fc.col("agent")["cached_input_tokens"].alias("cached_input_tokens"),
            fc.col("agent")["output_tokens"].alias("output_tokens"),
            fc.col("agent")["requests"].alias("requests"),
            fc.col("agent")["cost"].alias("cost"),
            fc.col("agent")["agent_usage_summary"].alias("agent_usage_summary"),
        )
        # Build scoring content and score
        .select(
            "*",
            fc.text.jinja(
                "<summary>\n{{ summary }}\n</summary>\n\n<feedback>\n{{ feedback }}\n</feedback>",
                summary=fc.col("summary"),
                feedback=fc.col("feedback"),
            ).alias("scoring_content"),
        )
        .select(
            "*",
            fc.semantic.map(
                f"{SCORING_PROMPT}\n\n<content>\n{{{{scoring_content}}}}\n</content>",
                scoring_content=fc.col("scoring_content"),
                response_format=FeedbackScore,
                strict=True,
                max_output_tokens=512,
            ).alias("score_struct"),
        )
        # Flatten score struct and create formatted strings
        .select(
            "*",
            format_2dp(fc.col("duration_raw")).alias("duration"),
            # numeric raw score for aggregation
            fc.col("score_struct")["overall"].alias("score_raw"),
            # formatted display fields
            format_2dp(fc.col("score_struct")["overall"]).alias("score"),
            format_1dp(fc.col("score_struct")["criteria"]["clarity"]).alias("clarity"),
            format_1dp(fc.col("score_struct")["criteria"]["actionability"]).alias("actionability"),
            format_1dp(fc.col("score_struct")["criteria"]["correctness"]).alias("correctness"),
            format_1dp(fc.col("score_struct")["criteria"]["coverage"]).alias("coverage"),
            format_1dp(fc.col("score_struct")["criteria"]["safety"]).alias("safety"),
        )
        # Build a combined source for cross-task reduction
        .select(
            "*",
            fc.text.jinja(
                """<summary>\n{{ summary }}\n</summary>\n\n<feedback>\n{{ feedback }}\n</feedback>""",
                summary=fc.col("summary"),
                feedback=fc.col("feedback"),
            ).alias("insight_source"),
        )
    )

    # Aggregate stats and render report in one go
    report_df = (
        pipeline_df
        .agg(
            fc.avg("duration_raw").alias("avg_duration"),
            fc.avg("num_tool_calls").alias("avg_tool_calls"),
            fc.sum("num_tool_calls").alias("total_tool_calls"),
            fc.avg("score_raw").alias("avg_score"),
            fc.count("prompt").alias("total_tasks"),
            # Totals for agent usage
            fc.sum("uncached_input_tokens").alias("total_uncached_input_tokens"),
            fc.sum("cached_input_tokens").alias("total_cached_input_tokens"),
            fc.sum("output_tokens").alias("total_output_tokens"),
            fc.sum("requests").alias("total_requests"),
            fc.sum("cost").alias("total_cost_raw"),
            # Cross-task reductions
            fc.semantic.reduce(
                "Summarize recurring themes, salient insights, and notable pitfalls across all tasks."
                " Use concise bullet points.",
                fc.col("insight_source"),
                max_output_tokens=1024,
                model_alias=evaluation_model,
            ).alias("salient_points"),
            fc.semantic.reduce(
                "From the <feedback> sections, compile action items grouped by priority: P0 (urgent/high-impact),"
                " P1 (important), P2 (nice-to-have). Reflect urgency/frequency indicated in feedback."
                " Output Markdown with headings P0/P1/P2 and bullet lists of one-line items with rationale.",
                fc.col("feedback"),
                max_output_tokens=1024,
                model_alias=evaluation_model,
            ).alias("action_items"),
            fc.collect_list(
                fc.struct(
                    fc.col("prompt"),
                    fc.col("response"),
                    fc.col("duration"),
                    fc.col("tool_calls_json").alias("tool_calls_json"),
                    fc.col("score"),
                    fc.col("clarity"),
                    fc.col("actionability"),
                    fc.col("correctness"),
                    fc.col("coverage"),
                    fc.col("safety"),
                    fc.col("summary"),
                    fc.col("feedback"),
                    # Per-task usage fields for template
                    fc.col("agent_usage_summary").alias("agent_usage_summary"),
                    fc.col("uncached_input_tokens").alias("uncached_input_tokens"),
                    fc.col("cached_input_tokens").alias("cached_input_tokens"),
                    fc.col("output_tokens").alias("output_tokens"),
                    fc.col("requests").alias("requests"),
                    fc.col("cost").alias("cost"),
                )
            ).alias("tasks"),
        )
        .select(
            fc.lit(agent_model).alias("agent_model"),
            fc.lit(evaluation_model).alias("evaluation_model"),
            format_2dp(fc.col("avg_duration")).alias("average_duration_s"),
            format_2dp(fc.col("avg_tool_calls")).alias("average_tool_calls"),
            fc.col("total_tool_calls"),
            format_2dp(fc.col("avg_score")).alias("average_score"),
            fc.col("total_tasks"),
            # Format totals
            fc.col("total_uncached_input_tokens"),
            fc.col("total_cached_input_tokens"),
            fc.col("total_output_tokens"),
            fc.col("total_requests"),
            format_2dp(fc.col("total_cost_raw")).alias("total_cost"),
            fc.col("salient_points"),
            fc.col("action_items"),
            fc.col("tasks"),
        )
        .select(
            fc.text.jinja(
                REPORT_JINJA_TEMPLATE,
                agent_model=fc.col("agent_model"),
                evaluation_model=fc.col("evaluation_model"),
                average_duration_s=fc.col("average_duration_s"),
                average_tool_calls=fc.col("average_tool_calls"),
                total_tool_calls=fc.col("total_tool_calls"),
                average_score=fc.col("average_score"),
                total_tasks=fc.col("total_tasks"),
                total_uncached_input_tokens=fc.col("total_uncached_input_tokens"),
                total_cached_input_tokens=fc.col("total_cached_input_tokens"),
                total_output_tokens=fc.col("total_output_tokens"),
                total_requests=fc.col("total_requests"),
                total_cost=fc.col("total_cost"),
                salient_points=fc.col("salient_points"),
                action_items=fc.col("action_items"),
                tasks=fc.col("tasks"),
            ).alias("report")
        )
    )

    # Extract the generated report
    report_result = report_df.to_pylist()
    report = report_result[0]["report"] if report_result else "Error generating report"
    
    session.stop()

    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate MCP servers")
    parser.add_argument(
        "--eval-file", required=True, help="Path to XML evaluation file"
    )
    parser.add_argument("--agent-model", default="openai/gpt-4o-mini", help="Model to use for the Agent calls")
    parser.add_argument("--evaluation-model", default="openai/gpt-4o-mini", help="Model to use for scoring (semantic.map)")
    parser.add_argument("--output", help="Output file for the report (default: stdout)")

    args = parser.parse_args()

    session = fc.Session.get_or_create(fc.SessionConfig(
        app_name="mcp_evaluation",
        semantic=fc.SemanticConfig(
            language_models={
                args.evaluation_model: OpenRouterLanguageModel(
                    model_name=args.evaluation_model,
                )
            }
        )
    ))
    # Run the evaluation
    report = asyncio.run(run_mcp_evaluation(args.eval_file, args.agent_model, args.evaluation_model, session))

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        logger.info(f"📄 Report saved to {args.output}")
    else:
        logger.info(report)


if __name__ == "__main__":
    main()
