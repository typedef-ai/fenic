# fenic.core.types.datatypes

Canonical HTML: https://docs.fenic.ai/latest/reference/fenic/core/types/datatypes/

Core data type definitions for the DataFrame API.

This module defines the type system used throughout the DataFrame API. It includes:
- Base classes for all data types
- Primitive types (string, integer, float, etc.)
- Composite types (arrays, structs)
- Specialized types (embeddings, markdown, etc.)

Classes:

- **`ArrayType`**
  –

  A type representing a homogeneous variable-length array (list) of elements.
- **`DataType`**
  –

  Base class for all data types.
- **`DocumentPathType`**
  –

  Represents a string containing a a document's local (file system) or remote (URL) path.
- **`EmbeddingType`**
  –

  A type representing a fixed-length embedding vector.
- **`StructField`**
  –

  A field in a StructType. Fields are nullable.
- **`StructType`**
  –

  A type representing a struct (record) with named fields.
- **`TranscriptType`**
  –

  Represents a string containing a transcript in a specific format.

Attributes:

- **`BooleanType`**
  –

  Represents a boolean value. (True/False)
- **`DateType`**
  –

  Represents a date value.
- **`DoubleType`**
  –

  Represents a 64-bit floating-point number.
- **`FloatType`**
  –

  Represents a 32-bit floating-point number.
- **`HtmlType`**
  –

  Represents a string containing raw HTML markup.
- **`IntegerType`**
  –

  Represents a signed integer value.
- **`JsonType`**
  –

  Represents a string containing JSON data.
- **`MarkdownType`**
  –

  Represents a string containing Markdown-formatted text.
- **`StringType`**
  –

  Represents a UTF-8 encoded string value.
- **`TimestampType`**
  –

  Represents a timestamp value.

## BooleanType

```
BooleanType = _BooleanType()
```

Represents a boolean value. (True/False)

## DateType

```
DateType = _DateType()
```

Represents a date value.

## DoubleType

```
DoubleType = _DoubleType()
```

Represents a 64-bit floating-point number.

## FloatType

```
FloatType = _FloatType()
```

Represents a 32-bit floating-point number.

## HtmlType

```
HtmlType = _HtmlType()
```

Represents a string containing raw HTML markup.

## IntegerType

```
IntegerType = _IntegerType()
```

Represents a signed integer value.

## JsonType

```
JsonType = _JsonType()
```

Represents a string containing JSON data.

## MarkdownType

```
MarkdownType = _MarkdownType()
```

Represents a string containing Markdown-formatted text.

## StringType

```
StringType = _StringType()
```

Represents a UTF-8 encoded string value.

## TimestampType

```
TimestampType = _TimestampType()
```

Represents a timestamp value.

## ArrayType

Bases: `DataType`

```
              flowchart TD
              fenic.core.types.datatypes.ArrayType[ArrayType]
              fenic.core.types.datatypes.DataType[DataType]

                              fenic.core.types.datatypes.DataType --> fenic.core.types.datatypes.ArrayType

              click fenic.core.types.datatypes.ArrayType href "" "fenic.core.types.datatypes.ArrayType"
              click fenic.core.types.datatypes.DataType href "" "fenic.core.types.datatypes.DataType"
```

A type representing a homogeneous variable-length array (list) of elements.

Attributes:

- **`element_type`**
  (`DataType`)
  –

  The data type of each element in the array.

Create an array of strings

```
ArrayType(StringType)
ArrayType(element_type=StringType)
```

## DataType

Bases: `ABC`

```
              flowchart TD
              fenic.core.types.datatypes.DataType[DataType]

              click fenic.core.types.datatypes.DataType href "" "fenic.core.types.datatypes.DataType"
```

Base class for all data types.

You won't instantiate this class directly. Instead, use one of the
concrete types like `StringType`, `ArrayType`, or `StructType`.

Used for casting, type validation, and schema inference in the DataFrame API.

## DocumentPathType

Bases: `_LogicalType`

```
              flowchart TD
              fenic.core.types.datatypes.DocumentPathType[DocumentPathType]
              fenic.core.types.datatypes._LogicalType[_LogicalType]
              fenic.core.types.datatypes.DataType[DataType]

                              fenic.core.types.datatypes._LogicalType --> fenic.core.types.datatypes.DocumentPathType
                                fenic.core.types.datatypes.DataType --> fenic.core.types.datatypes._LogicalType

              click fenic.core.types.datatypes.DocumentPathType href "" "fenic.core.types.datatypes.DocumentPathType"
              click fenic.core.types.datatypes._LogicalType href "" "fenic.core.types.datatypes._LogicalType"
              click fenic.core.types.datatypes.DataType href "" "fenic.core.types.datatypes.DataType"
```

Represents a string containing a a document's local (file system) or remote (URL) path.

## EmbeddingType

Bases: `_LogicalType`

```
              flowchart TD
              fenic.core.types.datatypes.EmbeddingType[EmbeddingType]
              fenic.core.types.datatypes._LogicalType[_LogicalType]
              fenic.core.types.datatypes.DataType[DataType]

                              fenic.core.types.datatypes._LogicalType --> fenic.core.types.datatypes.EmbeddingType
                                fenic.core.types.datatypes.DataType --> fenic.core.types.datatypes._LogicalType

              click fenic.core.types.datatypes.EmbeddingType href "" "fenic.core.types.datatypes.EmbeddingType"
              click fenic.core.types.datatypes._LogicalType href "" "fenic.core.types.datatypes._LogicalType"
              click fenic.core.types.datatypes.DataType href "" "fenic.core.types.datatypes.DataType"
```

A type representing a fixed-length embedding vector.

Attributes:

- **`dimensions`**
  (`int`)
  –

  The number of dimensions in the embedding vector.
- **`embedding_model`**
  (`str`)
  –

  Name of the model used to generate the embedding.

Create an embedding type for text-embedding-3-small

```
EmbeddingType(384, embedding_model="text-embedding-3-small")
```

## StructField

A field in a StructType. Fields are nullable.

Attributes:

- **`name`**
  (`str`)
  –

  The name of the field.
- **`data_type`**
  (`DataType`)
  –

  The data type of the field.

## StructType

Bases: `DataType`

```
              flowchart TD
              fenic.core.types.datatypes.StructType[StructType]
              fenic.core.types.datatypes.DataType[DataType]

                              fenic.core.types.datatypes.DataType --> fenic.core.types.datatypes.StructType

              click fenic.core.types.datatypes.StructType href "" "fenic.core.types.datatypes.StructType"
              click fenic.core.types.datatypes.DataType href "" "fenic.core.types.datatypes.DataType"
```

A type representing a struct (record) with named fields.

Attributes:

- **`fields`**
  –

  List of field definitions.

Create a struct with name and age fields

```
StructType([
    StructField("name", StringType),
    StructField("age", IntegerType),
])
```

## TranscriptType

Bases: `_LogicalType`

```
              flowchart TD
              fenic.core.types.datatypes.TranscriptType[TranscriptType]
              fenic.core.types.datatypes._LogicalType[_LogicalType]
              fenic.core.types.datatypes.DataType[DataType]

                              fenic.core.types.datatypes._LogicalType --> fenic.core.types.datatypes.TranscriptType
                                fenic.core.types.datatypes.DataType --> fenic.core.types.datatypes._LogicalType

              click fenic.core.types.datatypes.TranscriptType href "" "fenic.core.types.datatypes.TranscriptType"
              click fenic.core.types.datatypes._LogicalType href "" "fenic.core.types.datatypes._LogicalType"
              click fenic.core.types.datatypes.DataType href "" "fenic.core.types.datatypes.DataType"
```

Represents a string containing a transcript in a specific format.
