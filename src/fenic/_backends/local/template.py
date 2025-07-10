import json
from io import StringIO
from typing import Any, Dict, Optional

from fenic.core._logical_plan.expressions import EscapingRule, ParsedTemplateFormat


class TemplateFormatReader:
    """A simplified row-only parser for template formats."""

    def __init__(self, template_format: ParsedTemplateFormat, input_data: StringIO):
        self.format = template_format
        self.input = input_data
        self.finished = False

    def read_row(self) -> Optional[Dict[str, Any]]:
        """Read one row and apply the template format to it."""
        if self.finished or self.input.closed:
            return None

        try:
            row = self._parse_row()
            self._consume_newline()

            return row
        except EOFError:
            self.finished = True
            return None

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
        self._consume_delimiter(self.format.delimiters[-1])

        return row

    def _consume_delimiter(self, delimiter: str) -> bool:
        """Consume the expected delimiter from the stream."""
        if not delimiter:
            return True

        chunk = self._peek(len(delimiter))
        if chunk == delimiter:
            self.input.read(len(delimiter))  # Actually consume it
            return True
        return False

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
        """Read characters until the next delimiter or end of line."""
        next_delimiter = self._get_next_delimiter(field_index)

        if not next_delimiter:
            # Read until end of line
            return self._read_until_eol().strip()

        # Read until we find the specific delimiter
        chunks = []
        while True:
            if self._at_eol() or self._at_eof():
                break
            if self._peek(len(next_delimiter)) == next_delimiter:
                break
            chunks.append(self.input.read(1))

        return "".join(chunks).strip()

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
        if self.input.read(1) != '"':
            # Not properly quoted
            self.input.seek(self.input.tell() - 1)  # Put back the character
            return None

        chunks = []
        while True:
            char = self.input.read(1)
            if not char:  # EOF
                return None

            if char == '"':
                # Check for escaped quote
                if self._peek(1) == '"':
                    self.input.read(1)  # Consume the second quote
                    chunks.append('"')  # Add literal quote to result
                else:
                    # End of quoted field
                    break
            else:
                chunks.append(char)

        return "".join(chunks)

    def _get_next_delimiter(self, current_field_index: int) -> str:
        """Get the delimiter that should appear after the current field."""
        next_index = current_field_index + 1
        if next_index < len(self.format.delimiters):
            return self.format.delimiters[next_index]
        return ""

    def _read_until_eol(self) -> str:
        """Read characters until end of line or EOF."""
        chunks = []
        while True:
            if self._at_eol() or self._at_eof():
                break
            chunks.append(self.input.read(1))
        return "".join(chunks)

    def _consume_newline(self) -> None:
        """Consume a newline sequence (\\n or \\r\\n) if present."""
        if self._peek(1) == '\r':
            self.input.read(1)
            if self._peek(1) == '\n':
                self.input.read(1)
        elif self._peek(1) == '\n':
            self.input.read(1)

    def _peek(self, length: int) -> str:
        """Look ahead in the stream without consuming characters."""
        pos = self.input.tell()
        chunk = self.input.read(length)
        self.input.seek(pos)
        return chunk

    def _at_eol(self) -> bool:
        """Check if we're at end of line."""
        return self._peek(1) in ('\n', '\r')

    def _at_eof(self) -> bool:
        """Check if we're at end of file."""
        return self._peek(1) == ''
