# fenic.core.types.schema

Canonical HTML: https://docs.fenic.ai/latest/reference/fenic/core/types/schema/

Schema definitions for DataFrame structures.

This module provides classes for defining and working with DataFrame schemas.
It includes ColumnField for individual column definitions and Schema for complete
DataFrame structure definitions.

Classes:

- **`ColumnField`**
  –

  Represents a typed column in a DataFrame schema.
- **`DatasetMetadata`**
  –

  Metadata for a dataset (table or view).
- **`Schema`**
  –

  Represents the schema of a DataFrame.

## ColumnField

Represents a typed column in a DataFrame schema.

A ColumnField defines the structure of a single column by specifying its name
and data type. This is used as a building block for DataFrame schemas.

Attributes:

- **`name`**
  (`str`)
  –

  The name of the column.
- **`data_type`**
  (`DataType`)
  –

  The data type of the column, as a DataType instance.

## DatasetMetadata

Metadata for a dataset (table or view).

Attributes:

- **`schema`**
  (`Schema`)
  –

  The schema of the dataset.
- **`description`**
  (`Optional[str]`)
  –

  The natural language description of the dataset's contents.

## Schema

Represents the schema of a DataFrame.

A Schema defines the structure of a DataFrame by specifying an ordered collection
of column fields. Each column field defines the name and data type of a column
in the DataFrame.

Attributes:

- **`column_fields`**
  (`List[ColumnField]`)
  –

  An ordered list of ColumnField objects that define the
  structure of the DataFrame.

Methods:

- **`column_names`**
  –

  Get a list of all column names in the schema.

### column_names

```
column_names() -> List[str]
```

Get a list of all column names in the schema.

Returns:

- `List[str]`
  –

  A list of strings containing the names of all columns in the schema.

Source code in `src/fenic/core/types/schema.py`

```
def column_names(self) -> List[str]:
    """Get a list of all column names in the schema.

    Returns:
        A list of strings containing the names of all columns in the schema.
    """
    return [field.name for field in self.column_fields]
```
