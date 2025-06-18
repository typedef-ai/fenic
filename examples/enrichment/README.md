# Log Enrichment Pipeline

[View in Github](https://github.com/typedef-ai/fenic/blob/main/examples/enrichment/README.md)

A log processing system using fenic's text extraction and semantic enrichment capabilities to transform unstructured logs into actionable incident response data.

## Overview

This pipeline demonstrates log enrichment through multi-stage processing:

- Template-based parsing without regex
- Service metadata enrichment via joins
- LLM-powered error categorization and remediation
- Incident severity assessment with business context

## Prerequisites

1. Install fenic:

   ```bash
   pip install fenic
   ```

2. Configure OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

```bash
python enrichment.py
```

## Implementation

The pipeline processes logs through three stages:

1. **Parse**: Extract structured fields from syslog-format messages
2. **Enrich**: Join with service ownership and criticality data
3. **Analyze**: Apply LLM operations for incident response

### API Structure

```python
from fenic.api.session import Session, SessionConfig, SemanticConfig, OpenAIModelConfig
from fenic.api.functions import col, text, semantic
from pydantic import BaseModel, Field

# Configure session
config = SessionConfig(
    app_name="log_enrichment",
    semantic=SemanticConfig(
        language_models= {
            "mini" : OpenAIModelConfig(
            model_name="gpt-4o-mini",
            rpm=500,
            tpm=200_000
        )
        }
    )
)

# Define extraction schema with Pydantic
class ErrorAnalysis(BaseModel):
    error_category: str = Field(..., description="Main category of the error")
    affected_component: str = Field(..., description="Specific component affected")
    potential_cause: str = Field(..., description="Most likely root cause")

# Stage 1: Template extraction
parsed = logs_df.select(
    "raw_message", text.extract("${timestamp:none} [${level:none}] ${service:none}: ${message:none}")
)

# Stage 2: Metadata join
enriched = parsed.join(metadata_df, on="service", how="left")

# Stage 3: Semantic enrichment
final = enriched.select(
    semantic.extract("message", ErrorAnalysis).alias("analysis"),
    semantic.classify(
        text.concat(col("message"), lit(" (criticality: "), col("criticality"), lit(")")),
        ["low", "medium", "high", "critical"]
    ).alias("incident_severity"),
    semantic.map(
        "Generate remediation steps for: {message} | Service: {service} | Team: {team_owner}"
    ).alias("remediation_steps")
)
```

## Output Format

```shell
✅ Pipeline Complete! Final enriched logs:
----------------------------------------------------------------------
timestamp           level  service       message                  team_owner      error_category  incident_severity  remediation_steps
2024-01-15 14:32:01 ERROR  payment-api   Connection timeout...    payments-team   database        critical          1. Check Database Connectivity...
2024-01-15 14:32:15 WARN   user-service  Rate limit exceeded...   identity-team   resource        critical          1. Review Rate Limiting Config...

📈 Analytics Examples:

Error Category Distribution:
error_category    count
database          1
resource          5
authentication    4
network           5

High-Priority Incidents (Critical/High severity):
service           team_owner       incident_severity  on_call_channel    remediation_steps
payment-api       payments-team    critical          #payments-oncall   1. Check Database Connectivity...
user-service      identity-team    critical          #identity-alerts   1. Review Rate Limiting Config...
```

## Configuration

### Custom Log Templates

Parse different log formats:

```python
# Syslog format
log_template = "${timestamp:none} [${level:none}] ${service:none}: ${message:none}"

# Custom application format
log_template = "${service:none} | ${timestamp:none} | ${level:none} - ${message:none}"
```

## Troubleshooting

**Issue**: Template extraction returns empty fields  
**Solution**: Check template format matches log structure exactly, including spaces

**Issue**: Missing service metadata after join  
**Solution**: Use left join to preserve all logs; add default values for missing metadata

**Issue**: Generic remediation steps  
**Solution**: Include more context in semantic.map prompt (service, team, criticality)
