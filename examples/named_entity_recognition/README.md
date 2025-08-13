# Named Entity Recognition for Security Intelligence

<p>
  <a href="https://colab.research.google.com/github/typedef-ai/fenic/blob/main/examples/named_entity_recognition/ner.ipynb">
    <img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
  </a>
</p>

A security-focused NER pipeline using Fenic's semantic extraction capabilities to identify and analyze threats, vulnerabilities, and indicators of compromise from unstructured security reports.

## Overview

This pipeline demonstrates automated security entity extraction and risk assessment:

- Zero-shot entity extraction (CVEs, IPs, domains, hashes)
- Enhanced extraction with threat intelligence context
- Document chunking for comprehensive analysis
- Risk prioritization and actionable intelligence

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
python ner.py
```

## Implementation

The pipeline processes security reports through five stages:

1. **Basic NER**: Extract standard security entities
2. **Enhanced NER**: Add threat-specific context
3. **Chunking**: Handle long documents effectively
4. **Analytics**: Aggregate and analyze extracted entities
5. **Risk Assessment**: Generate actionable intelligence

## Troubleshooting

**Issue**: Incomplete entity extraction
**Solution**: Increase chunk size or adjust overlap percentage for better context

**Issue**: Missing threat actors or APT groups
**Solution**: Add more specific descriptions in the Pydantic field definitions

**Issue**: Generic risk assessments
**Solution**: Include more context about your organization in the assessment prompt
