# fenic.core.types.summarize

Canonical HTML: https://docs.fenic.ai/latest/reference/fenic/core/types/summarize/

Summary structures for different formats of summaries.

Classes:

- **`KeyPoints`**
  –

  Summary as a concise bulleted list.
- **`Paragraph`**
  –

  Summary as a cohesive narrative.

## KeyPoints

Bases: `BaseModel`

```
              flowchart TD
              fenic.core.types.summarize.KeyPoints[KeyPoints]

              click fenic.core.types.summarize.KeyPoints href "" "fenic.core.types.summarize.KeyPoints"
```

Summary as a concise bulleted list.

Each bullet should capture a distinct and essential idea, with a maximum number of points specified.

Attributes:

- **`max_points`**
  (`int`)
  –

  The maximum number of key points to include in the summary.

Methods:

- **`max_tokens`**
  –

  Calculate the maximum number of tokens for the summary based on the number of key points.

### max_tokens

```
max_tokens() -> int
```

Calculate the maximum number of tokens for the summary based on the number of key points.

Source code in `src/fenic/core/types/summarize.py`

```
def max_tokens(self) -> int:
    """Calculate the maximum number of tokens for the summary based on the number of key points."""
    return self.max_points * 75
```

## Paragraph

Bases: `BaseModel`

```
              flowchart TD
              fenic.core.types.summarize.Paragraph[Paragraph]

              click fenic.core.types.summarize.Paragraph href "" "fenic.core.types.summarize.Paragraph"
```

Summary as a cohesive narrative.

The summary should flow naturally and not exceed a specified maximum word count.

Attributes:

- **`max_words`**
  (`int`)
  –

  The maximum number of words allowed in the summary.

Methods:

- **`max_tokens`**
  –

  Calculate the maximum number of tokens for the summary based on the number of words.

### max_tokens

```
max_tokens() -> int
```

Calculate the maximum number of tokens for the summary based on the number of words.

Source code in `src/fenic/core/types/summarize.py`

```
def max_tokens(self) -> int:
    """Calculate the maximum number of tokens for the summary based on the number of words."""
    return int(self.max_words * 1.5)
```
