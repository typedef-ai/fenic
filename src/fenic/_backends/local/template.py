import json
from typing import Any, Dict, Optional

from fenic.core._logical_plan.expressions import EscapingRule, ParsedTemplateFormat


class TemplateFormatReader:
    """A parser for applying templates strings for structured extraction."""

    def __init__(self, template_format: ParsedTemplateFormat, input_string: str):
        self.format = template_format
        self.input_string = input_string
        self.position = 0

    def parse(self) -> Optional[Dict[str, Any]]:
        """Parse the input string using the template format."""
        try:
            return self._parse_row()
        except EOFError:
            return None  # Template didn't match the input

    def _parse_row(self) -> Dict[str, Any]:
        """Parse a single row according to the template format."""
        row = {}

        for i, col_name in enumerate(self.format.columns):
            # Match delimiter before this column
            if not self._consume_delimiter(self.format.delimiters[i]):
                raise EOFError("Failed to match delimiter")

            # Read the field value
            rule = self.format.escaping_rules[i]
            value = self._read_field(rule, i)
            if value is not None:
                row[col_name] = value

        # Match trailing delimiter (always exists)
        if not self._consume_delimiter(self.format.delimiters[-1]):
            raise EOFError("Failed to match final delimiter")

        return row

    def _read_field(self, rule: EscapingRule, field_index: int) -> Any:
        """Read a field value according to the escaping rule."""
        if rule == EscapingRule.NONE:
            return self._read_until_next_delimiter(field_index)
        elif rule == EscapingRule.CSV:
            return self._read_csv_field(field_index)
        elif rule == EscapingRule.JSON:
            return self._read_json_field(field_index)
        elif rule == EscapingRule.QUOTED:
            return self._read_quoted_field()
        else:
            raise ValueError(f"Unsupported escaping rule: {rule.name}")

    def _read_until_next_delimiter(self, field_index: int) -> str:
        """Read characters until the next delimiter or end of input."""
        next_delimiter = self._get_next_delimiter(field_index)

        if not next_delimiter:
            # Read until end of string (no more delimiters)
            result = self.input_string[self.position:]
            self.position = len(self.input_string)
            return result.strip()

        # Find the next occurrence of the delimiter
        delimiter_pos = self.input_string.find(next_delimiter, self.position)
        if delimiter_pos == -1:
            # Delimiter not found - read to end
            result = self.input_string[self.position:]
            self.position = len(self.input_string)
            return result.strip()

        # Read up to the delimiter
        result = self.input_string[self.position:delimiter_pos]
        self.position = delimiter_pos
        return result.strip()

    def _read_csv_field(self, field_index: int) -> str:
        """Read a CSV field (may be quoted or unquoted)."""
        if self._peek(1) == '"':
            return self._read_quoted_field()
        else:
            return self._read_until_next_delimiter(field_index).strip()

    def _read_json_field(self, field_index: int) -> Optional[str]:
        """Read and validate a JSON field."""
        text = self._read_until_next_delimiter(field_index).strip()
        if not text:
            return None

        try:
            json.loads(text)  # Validate JSON
            return text
        except json.JSONDecodeError:
            return None

    def _read_quoted_field(self) -> Optional[str]:
        """Read a quoted field with proper escape handling."""
        if self._peek(1) != '"':
            return None

        self.position += 1  # Skip opening quote
        chunks = []

        while self.position < len(self.input_string):
            char = self.input_string[self.position]

            if char == '"':
                # Check for escaped quote
                if self.position + 1 < len(self.input_string) and self.input_string[self.position + 1] == '"':
                    chunks.append('"')  # Add literal quote
                    self.position += 2  # Skip both quotes
                else:
                    # End of quoted field
                    self.position += 1  # Skip closing quote
                    break
            else:
                chunks.append(char)
                self.position += 1

        return "".join(chunks)

    def _get_next_delimiter(self, current_field_index: int) -> str:
        """Get the delimiter that should appear after the current field."""
        next_index = current_field_index + 1
        if next_index < len(self.format.delimiters):
            return self.format.delimiters[next_index]
        return ""

    def _consume_delimiter(self, delimiter: str) -> bool:
        """Consume the expected delimiter from the string."""
        if not delimiter:
            return True

        if self.input_string[self.position:].startswith(delimiter):
            self.position += len(delimiter)
            return True
        return False

    def _peek(self, length: int) -> str:
        """Look ahead in the string without advancing position."""
        return self.input_string[self.position:self.position + length]

    def _at_eof(self) -> bool:
        """Check if we're at end of string."""
        return self.position >= len(self.input_string)
