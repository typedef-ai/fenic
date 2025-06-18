# Markdown -> JSON Conversion

The `markdown.to_json()` function converts Markdown-formatted text into a hierarchical JSON representation that preserves the document's structure and formatting. This JSON format is optimized for document chunking, semantic analysis, and querying with tools like `jq`.

## Overview

The JSON representation follows a tree structure where each node represents a different Markdown element (headings, paragraphs, lists, etc.). The schema is designed to be both expressive and practical, making it easier to work with Markdown content programmatically.

## Key Features

- **Hierarchical Structure**: Content is organized into nested sections based on heading levels
- **Rich Formatting**: Preserves all Markdown formatting including inline styles, links, and code blocks
- **Metadata Support**: Optional document-level metadata for source, title, tags, and dates
- **Complete Markdown Support**: Handles all standard Markdown elements including:
  - Headings (h1-h6)
  - Paragraphs with inline formatting
  - Lists (ordered, unordered, task lists)
  - Tables with alignment
  - Code blocks with or without language info
  - Blockquotes
  - HTML blocks and inline elements

## Usage Examples

### Basic Conversion

```python
df.select(markdown.to_json(col("markdown_text")))
```

### Extract Headings with jq

```python
# Get all level-2 headings
df.select(json.jq(markdown.to_json(col("md")), ".. | select(.type == 'heading' and .level == 2)"))
```

## JSON Schema

The schema defines a tree structure where each node can be either a block-level element or an inline element.

<!-- markdownlint-disable MD033 -->
<details>

<summary>Raw JSON Schema</summary>

````json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Document",
  "type": "object",
  "required": ["type", "children"],
  "properties": {
    "type": {
      "const": "document"  // Root node type
    },
    "metadata": {  // Optional document metadata
      "type": "object",
      "properties": {
        "source": { "type": "string" },  // Source document path/URL
        "title": { "type": "string" },   // Document title
        "tags": {                        // Document tags
          "type": "array",
          "items": { "type": "string" }
        },
        "date": { "type": "string" }     // Document date
      },
      "additionalProperties": true
    },
    "children": {  // Main content blocks
      "type": "array",
      "items": { "$ref": "#/definitions/BlockNode" }
    }
  },
  "definitions": {
    "BlockNode": {  // Block-level elements
      "oneOf": [
        { "$ref": "#/definitions/Heading" },      // # Heading
        { "$ref": "#/definitions/Paragraph" },    // Regular paragraph
        { "$ref": "#/definitions/List" },         // Ordered/unordered list
        { "$ref": "#/definitions/Blockquote" },   // > Blockquote
        { "$ref": "#/definitions/CodeBlock" },    // ``` Code block
        { "$ref": "#/definitions/Table" },        // | Table |
        { "$ref": "#/definitions/ThematicBreak" },// ---
        { "$ref": "#/definitions/HtmlBlock" }     // HTML block
      ]
    },
    "Heading": {  // Heading structure
      "type": "object",
      "required": ["type", "level", "content", "children"],
      "properties": {
        "type": { "const": "heading" },
        "level": { "type": "integer", "minimum": 1, "maximum": 6 },  // h1-h6
        "content": {  // Heading text with inline formatting
          "type": "array",
          "items": { "$ref": "#/definitions/InlineNode" }
        },
        "children": {  // Content under this heading
          "type": "array",
          "items": { "$ref": "#/definitions/BlockNode" }
        }
      }
    },
    "Paragraph": {  // Paragraph structure
      "type": "object",
      "required": ["type", "content"],
      "properties": {
        "type": { "const": "paragraph" },
        "content": {  // Paragraph content with inline formatting
          "type": "array",
          "items": { "$ref": "#/definitions/InlineNode" }
        }
      }
    },
    "List": {  // List structure
      "type": "object",
      "required": ["type", "ordered", "tight", "items"],
      "properties": {
        "type": { "const": "list" },
        "ordered": { "type": "boolean" },  // true for ordered lists
        "start": { "type": "integer" },    // Starting number for ordered lists
        "tight": { "type": "boolean" },    // Whether list items are tightly packed
        "items": {  // List items
          "type": "array",
          "items": { "$ref": "#/definitions/ListItem" }
        }
      }
    },
    "ListItem": {  // List item structure
      "type": "object",
      "required": ["children"],
      "properties": {
        "checked": { "type": ["boolean", "null"] },  // For task lists
        "children": {  // Item content
          "type": "array",
          "items": { "$ref": "#/definitions/BlockNode" }
        }
      }
    },
    "CodeBlock": {  // Code block structure
      "type": "object",
      "required": ["type", "text", "fenced"],
      "properties": {
        "type": { "const": "code_block" },
        "info": { "type": "string" },      // Language info
        "language": { "type": "string" },  // Language identifier
        "text": { "type": "string" },      // Code content
        "fenced": { "type": "boolean" },   // Whether using ``` or indented
        "fence_char": { "type": "string" },// Fence character used
        "fence_length": { "type": "integer", "minimum": 1 }  // Number of fence chars
      }
    },
    "Table": {  // Table structure
      "type": "object",
      "required": ["type", "alignments", "header", "rows"],
      "properties": {
        "type": { "const": "table" },
        "alignments": {  // Column alignments
          "type": "array",
          "items": {
            "type": ["string", "null"],
            "enum": ["left", "center", "right", null]
          }
        },
        "header": {  // Table header
          "type": "array",
          "items": {
            "type": "array",
            "items": { "$ref": "#/definitions/InlineNode" }
          }
        },
        "rows": {  // Table rows
          "type": "array",
          "items": {
            "type": "array",
            "items": {
              "type": "array",
              "items": { "$ref": "#/definitions/InlineNode" }
            }
          }
        }
      }
    },
    "InlineNode": {  // Inline elements
      "oneOf": [
        { "$ref": "#/definitions/Text" },         // Plain text
        { "$ref": "#/definitions/Strong" },       // **Bold**
        { "$ref": "#/definitions/Emphasis" },     // *Italic*
        { "$ref": "#/definitions/Link" },         // [Link](url)
        { "$ref": "#/definitions/InlineCode" },   // `code`
        { "$ref": "#/definitions/Image" },        // ![Alt](src)
        { "$ref": "#/definitions/Strikethrough" },// ~~Strikethrough~~
        { "$ref": "#/definitions/HtmlInline" },   // <span>
        { "$ref": "#/definitions/HardBreak" }     // Line break
      ]
    }
  },
  "Text": {
      "type": "object",
      "required": ["type", "text"],
      "properties": {
        "type": { "const": "text" },
        "text": { "type": "string" }
      }
    },
    "Strong": {
      "type": "object",
      "required": ["type", "content"],
      "properties": {
        "type": { "const": "strong" },
        "content": {
          "type": "array",
          "items": { "$ref": "#/definitions/InlineNode" }
        }
      }
    },
    "Emphasis": {
      "type": "object",
      "required": ["type", "content"],
      "properties": {
        "type": { "const": "emphasis" },
        "content": {
          "type": "array",
          "items": { "$ref": "#/definitions/InlineNode" }
        }
      }
    },
    "Link": {
      "type": "object",
      "required": ["type", "href", "content"],
      "properties": {
        "type": { "const": "link" },
        "href": { "type": "string" },
        "title": { "type": "string" },
        "content": {
          "type": "array",
          "items": { "$ref": "#/definitions/InlineNode" }
        }
      }
    },
    "InlineCode": {
      "type": "object",
      "required": ["type", "text"],
      "properties": {
        "type": { "const": "inline_code" },
        "text": { "type": "string" }
      }
    },
    "Image": {
      "type": "object",
      "required": ["type", "src"],
      "properties": {
        "type": { "const": "image" },
        "src": { "type": "string" },
        "alt": { "type": "string" },
        "title": { "type": "string" }
      }
    },
    "Strikethrough": {
      "type": "object",
      "required": ["type", "content"],
      "properties": {
        "type": { "const": "strikethrough" },
        "content": {
          "type": "array",
          "items": { "$ref": "#/definitions/InlineNode" }
        }
      }
    },
    "HtmlInline": {
      "type": "object",
      "required": ["type", "text"],
      "properties": {
        "type": { "const": "html_inline" },
        "text": { "type": "string" }
      }
    },
    "HardBreak": {
      "type": "object",
      "required": ["type"],
      "properties": {
        "type": { "const": "hardbreak" }
      }
    }
  }
}
````

</details>

## Example: Markdown to JSON Conversion

Here's a simple example showing how a Markdown document is converted to JSON:

<details>

<summary>Raw Markdown</summary>
````markdown
# Project Documentation

This is a sample document with various Markdown elements.

## Features

- **Bold text** and _italic text_
- `Inline code` and [links](https://example.com)
- Code blocks with syntax highlighting:

```python
def hello():
    print("Hello, world!")
```

### Nested Section

> This is a blockquote with some text.

| Column 1 | Column 2 |
| -------- | -------- |
| Cell 1   | Cell 2   |

<!-- markdownlint-disable MD040 -->

````
</details>

The above Markdown is converted to this JSON structure:

<details>

<summary>Converted JSON</summary>
```json
{
   "type":"document",
   "children":[
      {
         "type":"heading",
         "level":1,
         "content":[
            {
               "type":"text",
               "text":"Project Documentation"
            }
         ],
         "children":[
            {
               "type":"paragraph",
               "content":[
                  {
                     "type":"text",
                     "text":"This is a sample document with various Markdown elements."
                  }
               ]
            },
            {
               "type":"heading",
               "level":2,
               "content":[
                  {
                     "type":"text",
                     "text":"Features"
                  }
               ],
               "children":[
                  {
                     "type":"list",
                     "ordered":false,
                     "tight":true,
                     "items":[
                        {
                           "children":[
                              {
                                 "type":"paragraph",
                                 "content":[
                                    {
                                       "type":"strong",
                                       "content":[
                                          {
                                             "type":"text",
                                             "text":"Bold text"
                                          }
                                       ]
                                    },
                                    {
                                       "type":"text",
                                       "text":" and "
                                    },
                                    {
                                       "type":"emphasis",
                                       "content":[
                                          {
                                             "type":"text",
                                             "text":"italic text"
                                          }
                                       ]
                                    }
                                 ]
                              }
                           ]
                        },
                        {
                           "children":[
                              {
                                 "type":"paragraph",
                                 "content":[
                                    {
                                       "type":"inline_code",
                                       "text":"Inline code"
                                    },
                                    {
                                       "type":"text",
                                       "text":" and "
                                    },
                                    {
                                       "type":"link",
                                       "href":"https://example.com",
                                       "content":[
                                          {
                                             "type":"text",
                                             "text":"links"
                                          }
                                       ]
                                    }
                                 ]
                              }
                           ]
                        },
                        {
                           "children":[
                              {
                                 "type":"paragraph",
                                 "content":[
                                    {
                                       "type":"text",
                                       "text":"Code blocks with syntax highlighting:"
                                    }
                                 ]
                              }
                           ]
                        }
                     ]
                  },
                  {
                     "type":"code_block",
                     "language":"python",
                     "text":"def hello():\n    print(\"Hello, world!\")",
                     "fenced":true,
                     "fence_char":"`",
                     "fence_length":3
                  },
                  {
                     "type":"heading",
                     "level":3,
                     "content":[
                        {
                           "type":"text",
                           "text":"Nested Section"
                        }
                     ],
                     "children":[
                        {
                           "type":"blockquote",
                           "children":[
                              {
                                 "type":"paragraph",
                                 "content":[
                                    {
                                       "type":"text",
                                       "text":"This is a blockquote with some text."
                                    }
                                 ]
                              }
                           ]
                        },
                        {
                           "type":"table",
                           "alignments":[
                              null,
                              null
                           ],
                           "header":[
                              [
                                 {
                                    "type":"text",
                                    "text":"Column 1"
                                 }
                              ],
                              [
                                 {
                                    "type":"text",
                                    "text":"Column 2"
                                 }
                              ]
                           ],
                           "rows":[
                              [
                                 [
                                    {
                                       "type":"text",
                                       "text":"Cell 1"
                                    }
                                 ],
                                 [
                                    {
                                       "type":"text",
                                       "text":"Cell 2"
                                    }
                                 ]
                              ]
                           ]
                        }
                     ]
                  }
               ]
            }
         ]
      }
   ]
}
```
</details>

This example demonstrates how:

1. The document structure is preserved in a tree format
2. Each Markdown element is represented as a specific node type
3. Inline formatting (bold, italic, code, links) is nested within their parent elements
4. Headings create a hierarchy with their content as children
5. Complex elements like tables and code blocks maintain their structure and metadata


## Common Use Cases

1. **Document Analysis**: Extract and analyze document structure
2. **Content Extraction**: Pull specific elements like code blocks or headings
3. **Document Transformation**: Convert between formats while preserving structure
4. **Semantic Search**: Build search indices based on document structure
5. **Content Validation**: Verify document structure and formatting

## Best Practices

1. Use `jq` queries to extract specific elements from the JSON structure
2. Leverage the hierarchical nature for document chunking
3. Use metadata for document organization and search
4. Consider the schema when building tools that process the JSON output
````
