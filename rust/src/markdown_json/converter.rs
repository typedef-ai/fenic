use crate::markdown_json::types::*;
use crate::markdown_json::MarkdownToJsonError;
use markdown::mdast::Node;
use markdown::{to_mdast, Constructs, ParseOptions};

pub struct MdAstConverter {
    heading_stack: Vec<Heading>,
    parse_options: ParseOptions,
}

impl Default for MdAstConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl MdAstConverter {
    pub fn new() -> Self {
        let parse_options = ParseOptions {
            constructs: Constructs {
                gfm_task_list_item: true,
                gfm_table: true,
                gfm_strikethrough: true,
                html_flow: true,
                html_text: true,
                ..Constructs::default()
            },
            ..ParseOptions::default()
        };

        Self {
            heading_stack: Vec::new(),
            parse_options,
        }
    }

    pub fn with_options(parse_options: ParseOptions) -> Self {
        Self {
            heading_stack: Vec::new(),
            parse_options,
        }
    }

    pub fn convert_markdown(&mut self, markdown: &str) -> Result<String, MarkdownToJsonError> {
        // Clear any state from previous conversions
        self.heading_stack.clear();

        // Parse markdown to AST
        let ast = to_mdast(markdown, &self.parse_options)
            .map_err(|e| MarkdownToJsonError::ParseError(e.to_string()))?;

        // Convert AST to Document
        let document = self.convert_root(ast)?;

        // Serialize to JSON string
        serde_json::to_string(&document).map_err(MarkdownToJsonError::SerdeError)
    }

    fn convert_root(&mut self, root: Node) -> Result<Document, MarkdownToJsonError> {
        if let Node::Root(root_node) = root {
            let mut document_children = Vec::new();

            for child in root_node.children {
                self.process_block_node(child, &mut document_children)?;
            }

            // Flush any remaining headings from the stack
            self.flush_headings(&mut document_children);

            Ok(Document {
                type_: "document".to_string(),
                metadata: None,
                children: document_children,
            })
        } else {
            Err(MarkdownToJsonError::StructureError(
                "Expected root node".to_string(),
            ))
        }
    }

    fn process_block_node(
        &mut self,
        node: Node,
        output: &mut Vec<BlockNode>,
    ) -> Result<(), MarkdownToJsonError> {
        match node {
            Node::Heading(heading) => {
                self.handle_heading(heading, output)?;
            }
            Node::Paragraph(paragraph) => {
                let block = self.convert_paragraph(paragraph)?;
                self.add_to_current_heading_or_output(block, output);
            }
            Node::List(list) => {
                let block = self.convert_list(list)?;
                self.add_to_current_heading_or_output(block, output);
            }
            Node::Blockquote(blockquote) => {
                let block = self.convert_blockquote(blockquote)?;
                self.add_to_current_heading_or_output(block, output);
            }
            Node::Code(code) => {
                let block = self.convert_code_block(code)?;
                self.add_to_current_heading_or_output(block, output);
            }
            Node::ThematicBreak(_) => {
                let block = BlockNode::ThematicBreak(ThematicBreak {
                    type_: "thematic_break".to_string(),
                });
                self.add_to_current_heading_or_output(block, output);
            }
            Node::Table(table) => {
                let block = self.convert_table(table)?;
                self.add_to_current_heading_or_output(block, output);
            }
            Node::Html(html) => {
                let block = BlockNode::HtmlBlock(HtmlBlock {
                    type_: "html_block".to_string(),
                    text: html.value,
                });
                self.add_to_current_heading_or_output(block, output);
            }
            // Handle other block nodes as needed - ignore unsupported types
            _ => {
                // Silently ignore unsupported node types
            }
        }
        Ok(())
    }

    fn handle_heading(
        &mut self,
        heading: markdown::mdast::Heading,
        output: &mut Vec<BlockNode>,
    ) -> Result<(), MarkdownToJsonError> {
        let level = heading.depth;

        // Pop headings deeper than or equal to current level
        while let Some(top_heading) = self.heading_stack.last() {
            if top_heading.level >= level {
                let completed_heading = self.heading_stack.pop().unwrap();
                self.add_completed_heading(completed_heading, output);
            } else {
                break;
            }
        }

        // Convert heading content to inline nodes
        let heading_content = Self::convert_inline_content(heading.children)?;

        // Create new heading
        let new_heading = Heading {
            type_: "heading".to_string(),
            level,
            content: heading_content,
            children: Vec::new(),
        };

        self.heading_stack.push(new_heading);
        Ok(())
    }

    fn add_to_current_heading_or_output(&mut self, block: BlockNode, output: &mut Vec<BlockNode>) {
        if let Some(current_heading) = self.heading_stack.last_mut() {
            current_heading.children.push(block);
        } else {
            output.push(block);
        }
    }

    fn add_completed_heading(&mut self, heading: Heading, output: &mut Vec<BlockNode>) {
        if let Some(parent_heading) = self.heading_stack.last_mut() {
            parent_heading.children.push(BlockNode::Heading(heading));
        } else {
            output.push(BlockNode::Heading(heading));
        }
    }

    fn flush_headings(&mut self, output: &mut Vec<BlockNode>) {
        while let Some(heading) = self.heading_stack.pop() {
            self.add_completed_heading(heading, output);
        }
    }

    fn convert_paragraph(
        &self,
        paragraph: markdown::mdast::Paragraph,
    ) -> Result<BlockNode, MarkdownToJsonError> {
        let content = Self::convert_inline_content(paragraph.children)?;
        Ok(BlockNode::Paragraph(Paragraph {
            type_: "paragraph".to_string(),
            content,
        }))
    }

    fn convert_list(&self, list: markdown::mdast::List) -> Result<BlockNode, MarkdownToJsonError> {
        let mut items = Vec::new();

        for item_node in list.children {
            if let Node::ListItem(list_item) = item_node {
                let mut item_children = Vec::new();

                for child in list_item.children {
                    match child {
                        Node::Paragraph(p) => {
                            item_children.push(self.convert_paragraph(p)?);
                        }
                        Node::List(nested_list) => {
                            item_children.push(self.convert_list(nested_list)?);
                        }
                        Node::Code(code) => {
                            item_children.push(self.convert_code_block(code)?);
                        }
                        Node::Blockquote(bq) => {
                            item_children.push(self.convert_blockquote(bq)?);
                        }
                        // Add other block types as needed - ignore unsupported types
                        _ => {
                            // Silently ignore unsupported node types
                        }
                    }
                }

                items.push(ListItem {
                    checked: list_item.checked,
                    children: item_children,
                });
            }
        }

        Ok(BlockNode::List(List {
            type_: "list".to_string(),
            ordered: list.ordered,
            start: list.start,
            tight: !list.spread, // tight is opposite of spread
            items,
        }))
    }

    fn convert_blockquote(
        &self,
        blockquote: markdown::mdast::Blockquote,
    ) -> Result<BlockNode, MarkdownToJsonError> {
        let mut children = Vec::new();

        for child in blockquote.children {
            match child {
                Node::Paragraph(p) => {
                    children.push(self.convert_paragraph(p)?);
                }
                Node::List(list) => {
                    children.push(self.convert_list(list)?);
                }
                Node::Code(code) => {
                    children.push(self.convert_code_block(code)?);
                }
                Node::Blockquote(nested_bq) => {
                    children.push(self.convert_blockquote(nested_bq)?);
                }
                // Add other block types as needed - ignore unsupported types
                _ => {
                    // Silently ignore unsupported node types
                }
            }
        }

        Ok(BlockNode::Blockquote(Blockquote {
            type_: "blockquote".to_string(),
            children,
        }))
    }

    fn convert_code_block(
        &self,
        code: markdown::mdast::Code,
    ) -> Result<BlockNode, MarkdownToJsonError> {
        let language = code.lang.clone();
        let info = code.meta.clone();

        Ok(BlockNode::CodeBlock(CodeBlock {
            type_: "code_block".to_string(),
            info,
            language,
            text: code.value,
            fenced: true,                      // mdast::Code is always fenced
            fence_char: Some("`".to_string()), // Default to backticks
            fence_length: Some(3),             // Default fence length
        }))
    }

    fn convert_table(
        &self,
        table: markdown::mdast::Table,
    ) -> Result<BlockNode, MarkdownToJsonError> {
        let alignments = table
            .align
            .into_iter()
            .map(|align| match align {
                markdown::mdast::AlignKind::Left => Some("left".to_string()),
                markdown::mdast::AlignKind::Center => Some("center".to_string()),
                markdown::mdast::AlignKind::Right => Some("right".to_string()),
                _ => None,
            })
            .collect();

        let mut header = Vec::new();
        let mut rows = Vec::new();

        for (i, row_node) in table.children.into_iter().enumerate() {
            if let Node::TableRow(row) = row_node {
                let mut row_cells = Vec::new();
                for cell_node in row.children {
                    if let Node::TableCell(cell) = cell_node {
                        let cell_content = Self::convert_inline_content(cell.children)?;
                        // Each cell is an array of inline nodes
                        row_cells.push(cell_content);
                    }
                }

                if i == 0 {
                    header = row_cells;
                } else {
                    rows.push(row_cells);
                }
            }
        }

        Ok(BlockNode::Table(Table {
            type_: "table".to_string(),
            alignments,
            header,
            rows,
        }))
    }

    fn convert_inline_content(children: Vec<Node>) -> Result<Vec<InlineNode>, MarkdownToJsonError> {
        let mut result = Vec::new();

        for child in children {
            match child {
                Node::Text(text) => {
                    result.push(InlineNode::Text(Text {
                        type_: "text".to_string(),
                        text: text.value,
                    }));
                }
                Node::Strong(strong) => {
                    let content = Self::convert_inline_content(strong.children)?;
                    result.push(InlineNode::Strong(Strong {
                        type_: "strong".to_string(),
                        content,
                    }));
                }
                Node::Emphasis(emphasis) => {
                    let content = Self::convert_inline_content(emphasis.children)?;
                    result.push(InlineNode::Emphasis(Emphasis {
                        type_: "emphasis".to_string(),
                        content,
                    }));
                }
                Node::InlineCode(code) => {
                    result.push(InlineNode::InlineCode(InlineCode {
                        type_: "inline_code".to_string(),
                        text: code.value,
                    }));
                }
                Node::Link(link) => {
                    let content = Self::convert_inline_content(link.children)?;
                    result.push(InlineNode::Link(Link {
                        type_: "link".to_string(),
                        href: link.url,
                        title: link.title,
                        content,
                    }));
                }
                Node::Image(image) => {
                    result.push(InlineNode::Image(Image {
                        type_: "image".to_string(),
                        src: image.url,
                        alt: Some(image.alt),
                        title: image.title,
                    }));
                }
                Node::Delete(delete) => {
                    let content = Self::convert_inline_content(delete.children)?;
                    result.push(InlineNode::Strikethrough(Strikethrough {
                        type_: "strikethrough".to_string(),
                        content,
                    }));
                }
                Node::Html(html) => {
                    result.push(InlineNode::HtmlInline(HtmlInline {
                        type_: "html_inline".to_string(),
                        text: html.value,
                    }));
                }
                Node::Break(_) => {
                    result.push(InlineNode::HardBreak(HardBreak {
                        type_: "hardbreak".to_string(),
                    }));
                }
                // Ignore unsupported inline node types
                _ => {
                    // Silently ignore unsupported node types
                }
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;
    use serde_json::Value;

    fn parse_json_output(result: Result<String, MarkdownToJsonError>) -> Value {
        let json_str = result.expect("Failed to convert markdown to JSON");
        serde_json::from_str(&json_str).expect("Failed to parse JSON output")
    }

    #[test]
    fn test_simple_paragraph() {
        let markdown = "This is a simple paragraph.";
        let mut converter = MdAstConverter::new();
        let result = converter.convert_markdown(markdown);
        let json = parse_json_output(result);

        assert_eq!(json["type"], "document");
        assert_eq!(json["children"].as_array().unwrap().len(), 1);

        let paragraph = &json["children"][0];
        assert_eq!(paragraph["type"], "paragraph");
        assert_eq!(paragraph["content"][0]["type"], "text");
        assert_eq!(
            paragraph["content"][0]["text"],
            "This is a simple paragraph."
        );
    }

    #[test]
    fn test_single_heading_with_content() {
        let markdown = "# Introduction\n\nThis is the introduction paragraph.";
        let mut converter = MdAstConverter::new();
        let result = converter.convert_markdown(markdown);
        let json = parse_json_output(result);

        assert_eq!(json["type"], "document");
        assert_eq!(json["children"].as_array().unwrap().len(), 1);

        let heading = &json["children"][0];
        assert_eq!(heading["type"], "heading");
        assert_eq!(heading["level"], 1);
        assert_eq!(heading["content"][0]["text"], "Introduction");
        assert_eq!(heading["children"].as_array().unwrap().len(), 1);

        let paragraph = &heading["children"][0];
        assert_eq!(paragraph["type"], "paragraph");
        assert_eq!(
            paragraph["content"][0]["text"],
            "This is the introduction paragraph."
        );
    }

    #[test]
    fn test_content_before_first_heading() {
        let markdown =
            "Intro paragraph before any headings.\n\n# First Section\n\nSection content.";
        let mut converter = MdAstConverter::new();
        let result = converter.convert_markdown(markdown);
        let json = parse_json_output(result);

        assert_eq!(json["children"].as_array().unwrap().len(), 2);

        // First child should be the paragraph before heading
        let intro = &json["children"][0];
        assert_eq!(intro["type"], "paragraph");
        assert_eq!(
            intro["content"][0]["text"],
            "Intro paragraph before any headings."
        );

        // Second child should be the heading
        let heading = &json["children"][1];
        assert_eq!(heading["type"], "heading");
        assert_eq!(heading["content"][0]["text"], "First Section");
    }

    #[test]
    fn test_nested_sections() {
        let markdown = indoc! {"
            # Main Section

            Main content.

            ## Subsection

            Subsection content.

            ### Deep Section

            Deep content.

            ## Another Subsection

            More content.
        "};

        let mut converter = MdAstConverter::new();
        let result = converter.convert_markdown(markdown);
        let json = parse_json_output(result);

        let main_heading = &json["children"][0];
        assert_eq!(main_heading["level"], 1);
        assert_eq!(main_heading["type"], "heading");
        assert_eq!(main_heading["content"][0]["text"], "Main Section");
        assert_eq!(main_heading["children"].as_array().unwrap().len(), 3);

        // First child: paragraph
        assert_eq!(main_heading["children"][0]["type"], "paragraph");

        // Second child: level 2 heading
        let subheading = &main_heading["children"][1];
        assert_eq!(subheading["type"], "heading");
        assert_eq!(subheading["level"], 2);
        assert_eq!(subheading["content"][0]["text"], "Subsection");
        assert_eq!(subheading["children"].as_array().unwrap().len(), 2);

        // Nested level 3 heading
        let deep_heading = &subheading["children"][1];
        assert_eq!(deep_heading["type"], "heading");
        assert_eq!(deep_heading["level"], 3);
        assert_eq!(deep_heading["content"][0]["text"], "Deep Section");

        // Third child: another level 2 heading
        let another_subheading = &main_heading["children"][2];
        assert_eq!(another_subheading["type"], "heading");
        assert_eq!(another_subheading["level"], 2);
        assert_eq!(
            another_subheading["content"][0]["text"],
            "Another Subsection"
        );
    }

    #[test]
    fn test_inline_formatting() {
        let markdown = "This has **bold**, *italic*, `code`, and [link](https://example.com) text.";
        let mut converter = MdAstConverter::new();
        let result = converter.convert_markdown(markdown);
        let json = parse_json_output(result);

        let content = &json["children"][0]["content"];
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "This has ");

        assert_eq!(content[1]["type"], "strong");
        assert_eq!(content[1]["content"][0]["text"], "bold");

        assert_eq!(content[2]["type"], "text");
        assert_eq!(content[2]["text"], ", ");

        assert_eq!(content[3]["type"], "emphasis");
        assert_eq!(content[3]["content"][0]["text"], "italic");

        assert_eq!(content[4]["type"], "text");
        assert_eq!(content[4]["text"], ", ");

        assert_eq!(content[5]["type"], "inline_code");
        assert_eq!(content[5]["text"], "code");

        assert_eq!(content[6]["type"], "text");
        assert_eq!(content[6]["text"], ", and ");

        assert_eq!(content[7]["type"], "link");
        assert_eq!(content[7]["href"], "https://example.com");
        assert_eq!(content[7]["content"][0]["text"], "link");
    }

    #[test]
    fn test_unordered_list() {
        let markdown = indoc! {"
            - First item
            - Second item
            - Third item
        "};

        let mut converter = MdAstConverter::new();
        let result = converter.convert_markdown(markdown);
        let json = parse_json_output(result);

        let list = &json["children"][0];
        assert_eq!(list["type"], "list");
        assert_eq!(list["ordered"], false);
        assert_eq!(list["tight"], true);
        assert_eq!(list["items"].as_array().unwrap().len(), 3);

        let first_item = &list["items"][0];
        assert_eq!(first_item["checked"], Value::Null);
        assert_eq!(
            first_item["children"][0]["content"][0]["text"],
            "First item"
        );
    }

    #[test]
    fn test_ordered_list() {
        let markdown = indoc! {"
            1. First item
            2. Second item
            3. Third item
        "};
        let mut converter = MdAstConverter::new();
        let result = converter.convert_markdown(markdown);
        let json = parse_json_output(result);

        let list = &json["children"][0];
        assert_eq!(list["type"], "list");
        assert_eq!(list["ordered"], true);
        assert_eq!(list["start"], 1);
        assert_eq!(list["items"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn test_task_list() {
        let markdown = indoc! {"
            - [x] Completed task
            - [ ] Incomplete task
            - [x] Another completed task
        "};

        let mut converter = MdAstConverter::new();
        let result = converter.convert_markdown(markdown);
        let json = parse_json_output(result);

        let list = &json["children"][0];
        let items = list["items"].as_array().unwrap();

        assert_eq!(items[0]["checked"], true);
        assert_eq!(items[1]["checked"], false);
        assert_eq!(items[2]["checked"], true);
    }

    #[test]
    fn test_nested_list() {
        let markdown = indoc! {"
            - Item 1
                - Nested item 1
                - Nested item 2
            - Item 2
        "};
        let mut converter = MdAstConverter::new();
        let result = converter.convert_markdown(markdown);
        let json = parse_json_output(result);

        let list = &json["children"][0];
        let first_item = &list["items"][0];
        assert_eq!(first_item["children"].as_array().unwrap().len(), 2);

        // First child is the paragraph, second is the nested list
        let nested_list = &first_item["children"][1];
        assert_eq!(nested_list["type"], "list");
        assert_eq!(nested_list["items"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_blockquote() {
        let markdown = "> This is a blockquote.\n> \n> With multiple paragraphs.";
        let mut converter = MdAstConverter::new();
        let result = converter.convert_markdown(markdown);
        let json = parse_json_output(result);

        let blockquote = &json["children"][0];
        assert_eq!(blockquote["type"], "blockquote");
        assert_eq!(blockquote["children"].as_array().unwrap().len(), 2);

        assert_eq!(blockquote["children"][0]["type"], "paragraph");
        assert_eq!(blockquote["children"][1]["type"], "paragraph");
    }

    #[test]
    fn test_code_block() {
        let markdown = indoc! {r#"
            ```rust
            fn main() {
                println!("Hello, world!");
            }
            ```
        "#};
        let mut converter = MdAstConverter::new();
        let result = converter.convert_markdown(markdown);
        let json = parse_json_output(result);

        let code_block = &json["children"][0];
        assert_eq!(code_block["type"], "code_block");
        assert_eq!(code_block["language"], "rust");
        assert_eq!(code_block["fenced"], true);
        assert_eq!(code_block["fence_char"], "`");
        assert_eq!(code_block["fence_length"], 3);
        assert!(code_block["text"].as_str().unwrap().contains("fn main()"));
    }

    #[test]
    fn test_thematic_break() {
        let markdown = "Before break\n\n---\n\nAfter break";
        let mut converter = MdAstConverter::new();
        let result = converter.convert_markdown(markdown);
        let json = parse_json_output(result);

        assert_eq!(json["children"].as_array().unwrap().len(), 3);
        assert_eq!(json["children"][0]["type"], "paragraph");
        assert_eq!(json["children"][1]["type"], "thematic_break");
        assert_eq!(json["children"][2]["type"], "paragraph");
    }

    #[test]
    fn test_complex_document() {
        let markdown = indoc! {r#"
            # Document Title

            Introduction paragraph with **bold** and *italic* text.

            ## Section One

            Content in section one.

            - List item 1
            - List item 2
              - Nested item

            > A blockquote in section one.

            ```python
            print("Code example")
            ```

            ## Section Two

            More content here.

            ### Subsection

            Deep nested content.

            ---

            Final paragraph after thematic break.
        "#};
        let mut converter = MdAstConverter::new();
        let result = converter.convert_markdown(markdown);
        let json = parse_json_output(result);

        // Should have one main heading
        assert_eq!(json["children"].as_array().unwrap().len(), 1);

        let main_heading = &json["children"][0];
        assert_eq!(main_heading["type"], "heading");
        assert_eq!(main_heading["content"][0]["text"], "Document Title");

        // Main heading should have: intro paragraph + 2 subheadings
        let main_children = main_heading["children"].as_array().unwrap();
        assert_eq!(main_children.len(), 3);

        // Check that we have the expected structure
        assert_eq!(main_children[0]["type"], "paragraph"); // intro
        assert_eq!(main_children[1]["type"], "heading"); // Section One
        assert_eq!(main_children[2]["type"], "heading"); // Section Two

        // Check Section One has the expected content
        let heading_one = &main_children[1];
        assert_eq!(heading_one["content"][0]["text"], "Section One");
        let heading_one_children = heading_one["children"].as_array().unwrap();
        assert_eq!(heading_one_children.len(), 4); // paragraph, list, blockquote, code

        // Check Section Two has subheading
        let heading_two = &main_children[2];
        assert_eq!(heading_two["content"][0]["text"], "Section Two");
        let heading_two_children = heading_two["children"].as_array().unwrap();
        assert_eq!(heading_two_children.len(), 2); // paragraph + subheading
        assert_eq!(heading_two_children[1]["type"], "heading");
        assert_eq!(heading_two_children[1]["content"][0]["text"], "Subsection");

        // Check that the subheading contains the thematic break and final paragraph
        let subheading = &heading_two_children[1];
        let subheading_children = subheading["children"].as_array().unwrap();
        assert_eq!(subheading_children.len(), 3); // paragraph + thematic_break + paragraph
        assert_eq!(subheading_children[0]["type"], "paragraph"); // "Deep nested content"
        assert_eq!(subheading_children[1]["type"], "thematic_break");
        assert_eq!(subheading_children[2]["type"], "paragraph"); // "Final paragraph"
        println!("json: {}", serde_json::to_string_pretty(&json).unwrap());
    }

    #[test]
    fn test_empty_document() {
        let markdown = "";
        let mut converter = MdAstConverter::new();
        let result = converter.convert_markdown(markdown);
        let json = parse_json_output(result);

        assert_eq!(json["type"], "document");
        assert_eq!(json["children"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn test_heading_with_inline_formatting() {
        let markdown = "# **Bold** *Italic* `Code` Heading";
        let mut converter = MdAstConverter::new();
        let result = converter.convert_markdown(markdown);
        let json = parse_json_output(result);

        let heading = &json["children"][0];
        assert_eq!(heading["type"], "heading");

        // Check that inline formatting is preserved
        let heading_content = &heading["content"];
        assert_eq!(heading_content[0]["type"], "strong");
        assert_eq!(heading_content[0]["content"][0]["text"], "Bold");
        assert_eq!(heading_content[1]["type"], "text");
        assert_eq!(heading_content[1]["text"], " ");
        assert_eq!(heading_content[2]["type"], "emphasis");
        assert_eq!(heading_content[2]["content"][0]["text"], "Italic");
        assert_eq!(heading_content[3]["type"], "text");
        assert_eq!(heading_content[3]["text"], " ");
        assert_eq!(heading_content[4]["type"], "inline_code");
        assert_eq!(heading_content[4]["text"], "Code");
        assert_eq!(heading_content[5]["type"], "text");
        assert_eq!(heading_content[5]["text"], " Heading");
    }

    #[test]
    fn test_multiple_top_level_sections() {
        let markdown = indoc! {"
            # Section A

            Content A.

            # Section B

            Content B.

            # Section C

            Content C.
        "};
        let mut converter = MdAstConverter::new();
        let result = converter.convert_markdown(markdown);
        let json = parse_json_output(result);

        // Should have 3 top-level headings
        assert_eq!(json["children"].as_array().unwrap().len(), 3);

        for (i, expected_title) in ["Section A", "Section B", "Section C"].iter().enumerate() {
            let heading = &json["children"][i];
            assert_eq!(heading["type"], "heading");
            assert_eq!(heading["level"], 1);
            assert_eq!(heading["content"][0]["text"], *expected_title);
            assert_eq!(heading["children"].as_array().unwrap().len(), 1); // one paragraph each
        }
    }

    #[test]
    fn test_table() {
        let markdown = indoc! {"
            | Name | Status | Score |
            |------|:------:|------:|
            | Alice | *Active* | 95.5 |
            | Bob | **Inactive** | 87 |
            | Charlie | `Pending` | 92.3 |
        "};

        let mut converter = MdAstConverter::new();
        let result = converter.convert_markdown(markdown);
        let json = parse_json_output(result);

        let table = &json["children"][0];
        assert_eq!(table["type"], "table");

        // Check alignments
        let alignments = table["alignments"].as_array().unwrap();
        assert_eq!(alignments.len(), 3);
        assert_eq!(alignments[0], Value::Null); // no alignment specified
        assert_eq!(alignments[1], "center");
        assert_eq!(alignments[2], "right");

        // Check header
        let header = table["header"].as_array().unwrap();
        assert_eq!(header.len(), 3); // 3 header cells
        assert_eq!(header[0][0]["type"], "text");
        assert_eq!(header[0][0]["text"], "Name");
        assert_eq!(header[1][0]["type"], "text");
        assert_eq!(header[1][0]["text"], "Status");
        assert_eq!(header[2][0]["type"], "text");
        assert_eq!(header[2][0]["text"], "Score");

        // Check rows
        let rows = table["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 3); // 3 data rows

        // First row: Alice
        let alice_row = &rows[0];
        assert_eq!(alice_row.as_array().unwrap().len(), 3); // 3 cells
        assert_eq!(alice_row[0][0]["type"], "text");
        assert_eq!(alice_row[0][0]["text"], "Alice");

        // Second cell should have italic formatting
        assert_eq!(alice_row[1][0]["type"], "emphasis");
        assert_eq!(alice_row[1][0]["content"][0]["text"], "Active");

        assert_eq!(alice_row[2][0]["type"], "text");
        assert_eq!(alice_row[2][0]["text"], "95.5");

        // Second row: Bob
        let bob_row = &rows[1];
        assert_eq!(bob_row[0][0]["text"], "Bob");

        // Second cell should have bold formatting
        assert_eq!(bob_row[1][0]["type"], "strong");
        assert_eq!(bob_row[1][0]["content"][0]["text"], "Inactive");

        assert_eq!(bob_row[2][0]["text"], "87");

        // Third row: Charlie
        let charlie_row = &rows[2];
        assert_eq!(charlie_row[0][0]["text"], "Charlie");

        // Second cell should have inline code formatting
        assert_eq!(charlie_row[1][0]["type"], "inline_code");
        assert_eq!(charlie_row[1][0]["text"], "Pending");

        assert_eq!(charlie_row[2][0]["text"], "92.3");
    }

    #[test]
    fn test_strikethrough() {
        let markdown = "This has ~~strikethrough~~ text.";
        let mut converter = MdAstConverter::new();
        let result = converter.convert_markdown(markdown);
        let json = parse_json_output(result);

        let content = &json["children"][0]["content"];
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "This has ");
        assert_eq!(content[1]["type"], "strikethrough");
        assert_eq!(content[1]["content"][0]["text"], "strikethrough");
        assert_eq!(content[2]["type"], "text");
        assert_eq!(content[2]["text"], " text.");
    }

    #[test]
    fn test_links_with_title() {
        let markdown = r#"Visit [example](https://example.com "Example Site") for more info."#;
        let mut converter = MdAstConverter::new();
        let result = converter.convert_markdown(markdown);
        let json = parse_json_output(result);

        let content = &json["children"][0]["content"];
        let link = &content[1];
        assert_eq!(link["type"], "link");
        assert_eq!(link["href"], "https://example.com");
        assert_eq!(link["title"], "Example Site");
        assert_eq!(link["content"][0]["text"], "example");
    }

    #[test]
    fn test_images_with_title() {
        let markdown = r#"![Alt text](image.png "Image Title")"#;
        let mut converter = MdAstConverter::new();
        let result = converter.convert_markdown(markdown);
        let json = parse_json_output(result);

        let content = &json["children"][0]["content"];
        let image = &content[0];
        assert_eq!(image["type"], "image");
        assert_eq!(image["src"], "image.png");
        assert_eq!(image["alt"], "Alt text");
        assert_eq!(image["title"], "Image Title");
    }

    #[test]
    fn test_html_inline() {
        let markdown = r#"This has <span class="highlight">HTML</span> content."#;
        let mut converter = MdAstConverter::new();
        let result = converter.convert_markdown(markdown);
        let json = parse_json_output(result);

        let content = &json["children"][0]["content"];
        assert_eq!(content.as_array().unwrap().len(), 5);

        // Check the structure: text, html_inline (open), text, html_inline (close), text
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "This has ");

        assert_eq!(content[1]["type"], "html_inline");
        assert_eq!(content[1]["text"], r#"<span class="highlight">"#);

        assert_eq!(content[2]["type"], "text");
        assert_eq!(content[2]["text"], "HTML");

        assert_eq!(content[3]["type"], "html_inline");
        assert_eq!(content[3]["text"], "</span>");

        assert_eq!(content[4]["type"], "text");
        assert_eq!(content[4]["text"], " content.");
    }

    #[test]
    fn test_html_block() {
        let markdown = indoc! {r#"
            <div class="custom">
              <p>HTML block content</p>
            </div>
        "#};
        let mut converter = MdAstConverter::new();
        let result = converter.convert_markdown(markdown);
        let json = parse_json_output(result);

        let html_block = &json["children"][0];
        assert_eq!(html_block["type"], "html_block");
        assert!(html_block["text"]
            .as_str()
            .unwrap()
            .contains("<div class=\"custom\">"));
    }

    #[test]
    fn test_nested_inline_formatting() {
        let markdown = "This is **bold with *italic* inside** text.";
        let mut converter = MdAstConverter::new();
        let result = converter.convert_markdown(markdown);
        let json = parse_json_output(result);

        let content = &json["children"][0]["content"];
        assert_eq!(content[1]["type"], "strong");
        let strong_content = &content[1]["content"];
        assert_eq!(strong_content[0]["type"], "text");
        assert_eq!(strong_content[0]["text"], "bold with ");
        assert_eq!(strong_content[1]["type"], "emphasis");
        assert_eq!(strong_content[1]["content"][0]["text"], "italic");
        assert_eq!(strong_content[2]["type"], "text");
        assert_eq!(strong_content[2]["text"], " inside");
    }
}
