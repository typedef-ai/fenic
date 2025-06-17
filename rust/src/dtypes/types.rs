use polars::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum FenicDType {
    // Primitive Types
    #[serde(rename = "StringType")]
    StringType,
    #[serde(rename = "IntegerType")]
    IntegerType,
    #[serde(rename = "FloatType")]
    FloatType,
    #[serde(rename = "DoubleType")]
    DoubleType,
    #[serde(rename = "BooleanType")]
    BooleanType,

    // Collection Types
    #[serde(rename = "ArrayType")]
    ArrayType { element_type: Box<FenicDType> },
    #[serde(rename = "StructType")]
    StructType { struct_fields: Vec<StructField> },

    // Semantic Types
    #[serde(rename = "EmbeddingType")]
    EmbeddingType {
        dimensions: usize,
        embedding_model: String,
    },
    #[serde(rename = "MarkdownType")]
    MarkdownType,
    #[serde(rename = "HtmlType")]
    HtmlType,
    #[serde(rename = "JsonType")]
    JsonType,
    #[serde(rename = "TranscriptType")]
    TranscriptType { format: Option<String> },
    #[serde(rename = "DocumentPathType")]
    DocumentPathType { format: Option<String> },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct StructField {
    pub name: String,
    #[serde(rename = "data_type")]
    pub data_type: Box<FenicDType>,
}

impl FenicDType {
    /// Returns the canonical Polars data type corresponding to this logical field type.
    ///
    /// This function maps the custom logical types to Polars `DataType`s,
    /// including recursive handling for nested array and struct types.
    pub fn canonical_polars_type(&self) -> DataType {
        match self {
            FenicDType::StringType => DataType::String,
            FenicDType::IntegerType => DataType::Int64,
            FenicDType::FloatType => DataType::Float32,
            FenicDType::DoubleType => DataType::Float64,
            FenicDType::BooleanType => DataType::Boolean,
            FenicDType::ArrayType { element_type } => {
                let element_type = element_type.canonical_polars_type();
                DataType::List(Box::new(element_type))
            }
            FenicDType::StructType { struct_fields } => {
                let fields = struct_fields
                    .iter()
                    .map(|field| {
                        Field::new(
                            field.name.as_str().into(),
                            field.data_type.canonical_polars_type(),
                        )
                    })
                    .collect();
                DataType::Struct(fields)
            }
            FenicDType::EmbeddingType {
                dimensions,
                embedding_model: _,
            } => DataType::Array(Box::new(DataType::Float32), *dimensions),
            FenicDType::JsonType => DataType::String,
            FenicDType::MarkdownType => DataType::String,
            FenicDType::HtmlType => DataType::String,
            FenicDType::TranscriptType { .. } => DataType::String,
            FenicDType::DocumentPathType { .. } => DataType::String,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fenic_dtype_serialization() {
        let dtype = FenicDType::StructType {
            struct_fields: vec![
                StructField {
                    name: "name".to_string(),
                    data_type: Box::new(FenicDType::StringType),
                },
                StructField {
                    name: "age".to_string(),
                    data_type: Box::new(FenicDType::IntegerType),
                },
            ],
        };

        let serialized = serde_json::to_string(&dtype).unwrap();
        let deserialized: FenicDType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(dtype, deserialized);
    }

    #[test]
    fn test_array_type_serialization() {
        let dtype = FenicDType::ArrayType {
            element_type: Box::new(FenicDType::IntegerType),
        };

        let serialized = serde_json::to_string(&dtype).unwrap();
        let deserialized: FenicDType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(dtype, deserialized);
    }

    #[test]
    fn test_canonical_polars_type_primitives() {
        assert_eq!(
            FenicDType::StringType.canonical_polars_type(),
            DataType::String
        );
        assert_eq!(
            FenicDType::IntegerType.canonical_polars_type(),
            DataType::Int64
        );
        assert_eq!(
            FenicDType::FloatType.canonical_polars_type(),
            DataType::Float32
        );
        assert_eq!(
            FenicDType::DoubleType.canonical_polars_type(),
            DataType::Float64
        );
        assert_eq!(
            FenicDType::BooleanType.canonical_polars_type(),
            DataType::Boolean
        );
        assert_eq!(
            FenicDType::JsonType.canonical_polars_type(),
            DataType::String
        );
    }

    #[test]
    fn test_canonical_polars_type_array() {
        let array_type = FenicDType::ArrayType {
            element_type: Box::new(FenicDType::StringType),
        };
        assert_eq!(
            array_type.canonical_polars_type(),
            DataType::List(Box::new(DataType::String))
        );
    }

    #[test]
    fn test_canonical_polars_type_struct() {
        let struct_type = FenicDType::StructType {
            struct_fields: vec![
                StructField {
                    name: "field1".to_string(),
                    data_type: Box::new(FenicDType::StringType),
                },
                StructField {
                    name: "field2".to_string(),
                    data_type: Box::new(FenicDType::IntegerType),
                },
            ],
        };

        match struct_type.canonical_polars_type() {
            DataType::Struct(fields) => {
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].name().as_str(), "field1");
                assert_eq!(fields[0].dtype(), &DataType::String);
                assert_eq!(fields[1].name().as_str(), "field2");
                assert_eq!(fields[1].dtype(), &DataType::Int64);
            }
            _ => panic!("Expected Struct type"),
        }
    }

    #[test]
    fn test_canonical_polars_type_embedding() {
        let embedding_type = FenicDType::EmbeddingType {
            dimensions: 128,
            embedding_model: "test-model".to_string(),
        };
        assert_eq!(
            embedding_type.canonical_polars_type(),
            DataType::Array(Box::new(DataType::Float32), 128)
        );
    }
}
