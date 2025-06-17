use serde::Serialize;

// Root document
#[derive(Serialize, Debug, Clone)]
pub struct Document {
    #[serde(rename = "type")]
    pub type_: String, // "document"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<DocumentMetadata>,
    pub children: Vec<BlockNode>,
}

#[derive(Serialize, Debug, Clone)]
pub struct DocumentMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub date: Option<String>,
    // Allow additional properties
    #[serde(flatten)]
    pub additional: std::collections::HashMap<String, serde_json::Value>,
}

// Block-level nodes
#[derive(Serialize, Debug, Clone)]
#[serde(untagged)]
pub enum BlockNode {
    Heading(Heading),
    Paragraph(Paragraph),
    List(List),
    Blockquote(Blockquote),
    CodeBlock(CodeBlock),
    Table(Table),
    ThematicBreak(ThematicBreak),
    HtmlBlock(HtmlBlock),
}

#[derive(Serialize, Debug, Clone)]
pub struct Heading {
    #[serde(rename = "type")]
    pub type_: String, // "heading"
    pub level: u8, // 1-6
    pub content: Vec<InlineNode>,
    pub children: Vec<BlockNode>,
}

#[derive(Serialize, Debug, Clone)]
pub struct Paragraph {
    #[serde(rename = "type")]
    pub type_: String, // "paragraph"
    pub content: Vec<InlineNode>,
}

#[derive(Serialize, Debug, Clone)]
pub struct List {
    #[serde(rename = "type")]
    pub type_: String, // "list"
    pub ordered: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start: Option<u32>,
    pub tight: bool,
    pub items: Vec<ListItem>,
}

#[derive(Serialize, Debug, Clone)]
pub struct ListItem {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checked: Option<bool>, // true, false, or null
    pub children: Vec<BlockNode>,
}

#[derive(Serialize, Debug, Clone)]
pub struct Blockquote {
    #[serde(rename = "type")]
    pub type_: String, // "blockquote"
    pub children: Vec<BlockNode>,
}

#[derive(Serialize, Debug, Clone)]
pub struct CodeBlock {
    #[serde(rename = "type")]
    pub type_: String, // "code_block"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub info: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    pub text: String,
    pub fenced: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fence_char: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fence_length: Option<u8>,
}

#[derive(Serialize, Debug, Clone)]
pub struct Table {
    #[serde(rename = "type")]
    pub type_: String, // "table"
    pub alignments: Vec<Option<String>>, // "left", "center", "right", or null
    pub header: Vec<Vec<InlineNode>>,
    pub rows: Vec<Vec<Vec<InlineNode>>>,
}

#[derive(Serialize, Debug, Clone)]
pub struct ThematicBreak {
    #[serde(rename = "type")]
    pub type_: String, // "thematic_break"
}

#[derive(Serialize, Debug, Clone)]
pub struct HtmlBlock {
    #[serde(rename = "type")]
    pub type_: String, // "html_block"
    pub text: String,
}

// Inline nodes
#[derive(Serialize, Debug, Clone)]
#[serde(untagged)]
pub enum InlineNode {
    Text(Text),
    Strong(Strong),
    Emphasis(Emphasis),
    Link(Link),
    InlineCode(InlineCode),
    Image(Image),
    Strikethrough(Strikethrough),
    HtmlInline(HtmlInline),
    HardBreak(HardBreak),
}

#[derive(Serialize, Debug, Clone)]
pub struct Text {
    #[serde(rename = "type")]
    pub type_: String, // "text"
    pub text: String,
}

#[derive(Serialize, Debug, Clone)]
pub struct Strong {
    #[serde(rename = "type")]
    pub type_: String, // "strong"
    pub content: Vec<InlineNode>,
}

#[derive(Serialize, Debug, Clone)]
pub struct Emphasis {
    #[serde(rename = "type")]
    pub type_: String, // "emphasis"
    pub content: Vec<InlineNode>,
}

#[derive(Serialize, Debug, Clone)]
pub struct Link {
    #[serde(rename = "type")]
    pub type_: String, // "link"
    pub href: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub content: Vec<InlineNode>,
}

#[derive(Serialize, Debug, Clone)]
pub struct InlineCode {
    #[serde(rename = "type")]
    pub type_: String, // "inline_code"
    pub text: String,
}

#[derive(Serialize, Debug, Clone)]
pub struct Image {
    #[serde(rename = "type")]
    pub type_: String, // "image"
    pub src: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

#[derive(Serialize, Debug, Clone)]
pub struct Strikethrough {
    #[serde(rename = "type")]
    pub type_: String, // "strikethrough"
    pub content: Vec<InlineNode>,
}

#[derive(Serialize, Debug, Clone)]
pub struct HtmlInline {
    #[serde(rename = "type")]
    pub type_: String, // "html_inline"
    pub text: String,
}

#[derive(Serialize, Debug, Clone)]
pub struct HardBreak {
    #[serde(rename = "type")]
    pub type_: String, // "hardbreak"
}
