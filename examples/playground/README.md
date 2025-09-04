# Fenic Playground (No API Key)

This playground lets you try out **fenic-style workflows locally without any API key**.  
It uses a lightweight `MockLLM` that returns canned responses so you can explore core ideas like:

- Async UDF-style processing with batching (`asyncio.gather`)  
- `semantic_extract` producing structured output validated by **Pydantic**  
- A mocked `semantic_join` to illustrate similarity-based joins  
- A minimal DataFrame-like wrapper (`MiniDF`) to mimic the fenic API feel  

---


## Requirements
Install dependencies first (in a virtual environment):

```bash
pip install -e ".[dev]"


## Quick start

From the repository root, run:

```bash
python examples/playground/playground.py
