# Meeting Transcript Processing with Fenic

[View in Github](https://github.com/typedef-ai/fenic/blob/main/examples/meeting_transcript_processing/README.md)

This example demonstrates how to use Fenic to automatically extract actionable insights from engineering meeting transcripts using semantic extraction and structured data processing.

## Overview

Engineering teams generate valuable knowledge in meetings, but capturing and organizing this information is often manual and error-prone. This pipeline automates the extraction of:

- **Action Items**: Tasks, assignees, and deadlines
- **Decisions**: Key decisions and their rationale
- **Technical Entities**: Services, technologies, metrics, and incident references
- **Team Analytics**: Workload distribution and productivity metrics

## Features

- **Native transcript parsing** with fenic's built-in functions
- **Semantic extraction** of technical entities, action items, and decisions
- **Structured data processing** on unstructured meeting content
- **Automated knowledge capture** for engineering teams
- **Actionable insights** for project management and team coordination

## Sample Data

The example processes three types of engineering meetings:

1. **Architecture Review** - Technical discussions about system design and bottlenecks
2. **Incident Post-Mortem** - Analysis of outages and mitigation strategies
3. **Sprint Planning** - Task allocation and project prioritization

## Pipeline Steps

### Step 1: Transcript Parsing

```python
# Parse transcripts into structured segments
parsed_transcripts_df = transcripts_df.with_column(
    "structured_transcript",
    fc.text.parse_transcript(fc.col("transcript"), 'generic')
)
```

### Step 2: Segment Extraction

Break down transcripts into individual speaking segments with speaker, start_time, and content.

### Step 3: Semantic Schema Definition

Define extraction schemas using Pydantic models:

```python
class TechnicalEntitiesSchema(BaseModel):
    services: str = Field(description="Technical services or systems mentioned")
    # ... more fields

# Action items using Pydantic
class ActionItemSchema(BaseModel):
    has_action_item: str = Field(description="Whether this segment contains an action item (yes/no)")
    assignee: str = Field(default=None, description="Person assigned to the action item")
    task_description: str = Field(description="Description of the task or action")
    deadline: str = Field(default=None, description="When the task should be completed")
```

### Step 4: Semantic Extraction

Apply AI-powered extraction to identify structured information from natural language:

```python
enriched_df = segments_df.with_column(
    "technical_entities",
    fc.semantic.extract(fc.col("content"), TechnicalEntitiesSchema)
).with_column(
    "action_items",
    fc.semantic.extract(fc.col("content"), ActionItemSchema)
).with_column(
    "decisions",
    fc.semantic.extract(fc.col("content"), DecisionSchema)
)
```

### Step 5: Analytics and Aggregation

Generate meeting-level insights and team analytics:

- Action item workload by team member
- Technology and service mentions across meetings
- Decision summary and rationale tracking
- Meeting productivity metrics

## Expected Output

The pipeline produces structured insights including:

**Action Items Summary:**

| meeting_id  | meeting_type         | assignee | task_description        | deadline     |
| ----------- | -------------------- | -------- | ----------------------- | ------------ |
| ARCH-2024-1 | Architecture Review  | Mike     | investigate Redis impl  | next Friday  |
| INC-2024-12 | Incident Post-Mortem | Sam      | review batch processing | tomorrow EOD |

**Team Workload Distribution:**

| assignee | assigned_tasks |
| -------- | -------------- |
| Mike     | 2              |
| Sam      | 1              |
| Lisa     | 1              |

**Technology Mentions:**

| technologies | mention_count |
| ------------ | ------------- |
| Redis        | 3             |
| PostgreSQL   | 2             |
| JWT          | 2             |

## Prerequisites

1. **OpenAI API Key**: Required for semantic extraction

   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. **Fenic Installation**:
   ```bash
   uv sync
   uv run maturin develop --uv
   ```

## Running the Example

```bash
uv run python examples/meeting_transcript_processing/transcript_processing.py
```

## Use Cases

This pipeline is valuable for:

- **Engineering Managers**: Track team workload and action item distribution
- **Technical Program Managers**: Monitor project decisions and technical debt
- **DevOps Teams**: Analyze incident patterns and response procedures
- **Architecture Teams**: Identify technology adoption trends and system bottlenecks

## Extensions

The example can be extended to:

- Integrate with calendar systems for automatic transcript ingestion
- Export to project management tools (Jira, Linear, etc.)
- Build dashboards for engineering metrics
- Create automated follow-up reminders
- Analyze team communication patterns

## Technical Notes

- Uses `gpt-4o-mini` for fast and cost-effective semantic extraction
- Handles mixed transcript formats automatically
- Implements workarounds for current framework limitations
