# Hello World

[View in Github](https://github.com/typedef-ai/fenic/blob/main/examples/hello_world/README.md)

An error log analyzer using Fenic's semantic extraction capabilities to parse and analyze application errors without regex patterns.

## Overview

This tool demonstrates automated error log analysis through natural language processing, providing:

- Root cause identification
- Automated fix suggestions
- Severity classification (low/medium/high/critical)
- Pattern extraction

## Prerequisites

1. Install Fenic:

   ```bash
   pip install fenic
   ```

2. Configure OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

```bash
python hello_world.py
```

## Implementation

The analyzer processes various error types including:

- Java NullPointerException
- Node.js connection errors (ECONNREFUSED)
- Python API timeouts (Stripe APIConnectionError)
- Database connection failures (Django OperationalError)
- React TypeError
- Performance warnings (slow queries)
- Cache misses and email delivery delays

## Troubleshooting

**Issue**: Generic analysis results
**Solution**: Add more descriptive fields to ExtractSchema

**Issue**: Incorrect severity classification
**Solution**: Adjust classification categories or provide examples

**Issue**: Missing error patterns
**Solution**: Modify ExtractSchemaField descriptions for better targeting
