import json
from io import StringIO
from typing import Any, Dict, Iterator, Optional

from fenic.core._logical_plan.expressions import EscapingRule, ParsedTemplateFormat


class TemplateFormatReader:
    """A simplified row-only parser. It does not handle a separate result-set prefix/suffix.
    Reads lines from input, expects columns as defined in ParsedTemplateFormat.
    """

    def __init__(self, template_format: ParsedTemplateFormat, input_data: StringIO):
        self.format = template_format
        self.input = input_data
        self.row_num = 0
        self.finished = False
        self.current_field_index = 0  # Track which field we're currently reading

    def read_row(self) -> Optional[Dict[str, Any]]:
        if self.finished or self.input.closed:
            return None

        row = {}
        self.current_field_index = 0  # Reset for each row

        try:
            for i, col_name in enumerate(self.format.columns):
                self.current_field_index = i

                # Match the delimiter before this column:
                if not self._match_delimiter(self.format.delimiters[i]):
                    # If we fail to match, presumably we're out of data:
                    raise EOFError()

                # Read the field
                rule = self.format.escaping_rules[i]
                value = self._read_field(rule)
                if value is not None:
                    row[col_name] = value

            # Finally match trailing delimiter (after last column):
            if len(self.format.delimiters) > len(self.format.columns):
                tail_delim = self.format.delimiters[len(self.format.columns)]
                if not self._match_delimiter(tail_delim):
                    # If the last delimiter doesn't match, treat it as EOF / end of row
                    pass

            # Attempt to consume one newline so the next row can start fresh:
            self._skip_newline()

            self.row_num += 1
            return row

        except EOFError:
            self.finished = True
            return None

    def read_all(self) -> Iterator[Dict[str, Any]]:
        while True:
            row = self.read_row()
            if row is None:
                break
            yield row

    def _match_delimiter(self, delimiter: str) -> bool:
        """If delimiter is empty, treat it as a no-op. Otherwise read from stream and compare."""
        if delimiter == "":
            return True

        pos = self.input.tell()
        chunk = self.input.read(len(delimiter))

        if chunk == delimiter:
            return True
        else:
            # revert
            self.input.seek(pos)
            return False

    def _skip_newline(self) -> None:
        r"""Consume one newline (\n or \r\n) if present."""
        pos = self.input.tell()
        c = self.input.read(1)
        if not c:
            return  # EOF
        if c == "\r":
            nxt = self.input.read(1)
            if nxt != "\n":
                # Revert the extra char
                self.input.seek(self.input.tell() - 1)
        elif c != "\n":
            # Not a newline, revert
            self.input.seek(pos)

    def _read_field(self, rule: EscapingRule) -> Any:
        if rule == EscapingRule.NONE:
            return self._read_until_next_delimiter()
        elif rule == EscapingRule.CSV:
            return self._read_csv_field()
        elif rule == EscapingRule.JSON:
            return self._read_json_field()
        elif rule == EscapingRule.QUOTED:
            return self._read_quoted_field()
        else:
            raise ValueError(f"Unsupported rule: {rule.name}")

    def _read_until_next_delimiter(self) -> str:
        """Read characters until we see the NEXT expected delimiter or newline."""
        # Get the next expected delimiter (after current field)
        next_delimiter_index = self.current_field_index + 1

        # If we're at the last field, read until end of line or EOF
        if next_delimiter_index >= len(self.format.delimiters):
            chunks = []
            while True:
                pos = self.input.tell()
                c = self.input.read(1)
                if not c:  # EOF
                    break
                if c in ('\n', '\r'):
                    # Put back the newline for _skip_newline to handle
                    self.input.seek(pos)
                    break
                chunks.append(c)
            return "".join(chunks).strip()

        next_delimiter = self.format.delimiters[next_delimiter_index]

        # If next delimiter is empty, read until newline or EOF
        if not next_delimiter:
            chunks = []
            while True:
                pos = self.input.tell()
                c = self.input.read(1)
                if not c:  # EOF
                    break
                if c in ('\n', '\r'):
                    # Put back the newline
                    self.input.seek(pos)
                    break
                chunks.append(c)
            return "".join(chunks).strip()

        # Read until we find the specific next delimiter, newline, or EOF
        chunks = []
        while True:
            pos = self.input.tell()
            c = self.input.read(1)
            if not c:  # EOF
                break

            # Put back the char so we can check if it matches the delimiter
            self.input.seek(pos)

            # Check if we hit a newline (end of line)
            if self._check_string("\n") or self._check_string("\r\n"):
                break

            # Check if we hit the next expected delimiter
            if self._check_string(next_delimiter):
                break

            # Otherwise consume the character
            c = self.input.read(1)
            chunks.append(c)

        return "".join(chunks).strip()

    def _read_csv_field(self) -> str:
        # For CSV fields, we need to handle quoted content specially
        # Check if the field starts with a quote
        pos = self.input.tell()
        first_char = self.input.read(1)

        if first_char == '"':
            # This is a quoted CSV field, read until closing quote
            self.input.seek(pos)  # Reset to start of field
            return self._read_quoted_field()
        else:
            # Not quoted, revert and read normally
            self.input.seek(pos)
            text = self._read_until_next_delimiter()
            return text.strip()

    def _read_json_field(self) -> str:
        text = self._read_until_next_delimiter()
        text = text.strip()
        if not text:
            return None
        try:
            # Validate that it's valid JSON by parsing it
            json.loads(text)
            # But return the original string, not the parsed object
            return text
        except json.JSONDecodeError:
            return None

    def _read_quoted_field(self) -> Optional[str]:
        """Read a quoted field, return None if not properly quoted or malformed.
        Expects an opening quote, reads until closing quote.
        Double quotes ("") are escaped to single quotes (").
        """
        # Check we actually have a leading quote
        start = self.input.read(1)
        if start != '"':
            if start:  # revert that char
                self.input.seek(self.input.tell() - 1)
            return None  # Not quoted - return None instead of error

        chunks = []
        try:
            while True:
                c = self.input.read(1)
                if not c:
                    # EOF in quoted field - return None instead of error
                    return None

                if c == '"':
                    # Could be end of field or doubled quote
                    pos = self.input.tell()
                    nxt = self.input.read(1)
                    if nxt != '"':
                        # Not a doubled quote => end of field, revert the extra char
                        self.input.seek(pos)
                        break
                    else:
                        # It's a double quote => literal " in the value
                        chunks.append('"')
                else:
                    chunks.append(c)

            return "".join(chunks)

        except Exception:
            # Any other parsing error - return None gracefully
            return None

    def _check_string(self, s: str) -> bool:
        """Look ahead to see if the next bytes match `s`. If not, revert."""
        pos = self.input.tell()
        chunk = self.input.read(len(s))
        if chunk == s:
            # revert pointer, only a peek
            self.input.seek(pos)
            return True
        else:
            self.input.seek(pos)
            return False
