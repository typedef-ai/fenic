# fenic.core.mcp

Canonical HTML: https://docs.fenic.ai/latest/reference/fenic/core/mcp/

MCP Tool Generation/FastMCP Server Management.

Classes:

- **`BoundToolParam`**
  –

  A bound tool parameter.
- **`SystemTool`**
  –

  A tool implemented as a regular Python function with explicit parameters.
- **`ToolParam`**
  –

  A parameter for a parameterized view tool.
- **`UserDefinedTool`**
  –

  A tool that has been bound to a specific Parameterized View.

## BoundToolParam

A bound tool parameter.

A bound tool parameter is a parameter that has been bound to a specific, typed,
`tool_param` usage within a Dataframe.

## SystemTool

A tool implemented as a regular Python function with explicit parameters.

The function must be a `Callable[..., LogicalPlan]`
(a function defined with `async def`). Collection/formatting is handled by
the MCP generator wrapper.

## ToolParam

Bases: `BaseModel`

```
              flowchart TD
              fenic.core.mcp.ToolParam[ToolParam]

              click fenic.core.mcp.ToolParam href "" "fenic.core.mcp.ToolParam"
```

A parameter for a parameterized view tool.

A parameter is a named value that can be passed to a tool. These are matched to the
parameter names of the `tool_param` UnresolvedLiteralExpr expressions captured in the Logical Plan.

Attributes:

- **`name`**
  (`str`)
  –

  The name of the parameter.
- **`description`**
  (`str`)
  –

  The description of the parameter.
- **`allowed_values`**
  (`Optional[List[ToolParameterType]]`)
  –

  The allowed values for the parameter.
- **`has_default`**
  (`bool`)
  –

  Whether the parameter has a default value.
- **`default_value`**
  (`Optional[ToolParameterType]`)
  –

  The default value for the parameter.

### required

```
required: bool
```

Whether the parameter is required.

Returns:

- `bool`
  –

  True if the parameter is required, False otherwise.

## UserDefinedTool

A tool that has been bound to a specific Parameterized View.
