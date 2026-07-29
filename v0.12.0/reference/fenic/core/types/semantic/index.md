# fenic.core.types.semantic

Canonical HTML: https://docs.fenic.ai/latest/reference/fenic/core/types/semantic/

Types used to configure model selection for semantic functions.

Classes:

- **`ModelAlias`**
  –

  A combination of a model name and a required profile for that model.

## ModelAlias

Bases: `BaseModel`

```
              flowchart TD
              fenic.core.types.semantic.ModelAlias[ModelAlias]

              click fenic.core.types.semantic.ModelAlias href "" "fenic.core.types.semantic.ModelAlias"
```

A combination of a model name and a required profile for that model.

Model aliases are used to select a specific model to use in a semantic operation.
Both the model name and profile must be specified when creating a ModelAlias.

Attributes:

- **`name`**
  (`str`)
  –

  The name of the model.
- **`profile`**
  (`str`)
  –

  The name of a profile configuration to use for the model.

Example

```
model_alias = ModelAlias(name="o4-mini", profile="low")
```
