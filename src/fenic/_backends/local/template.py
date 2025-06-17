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

    def read_row(self) -> Optional[Dict[str, Any]]:
        if self.finished or self.input.closed:
            return None

        row = {}
        try:
            for i, col_name in enumerate(self.format.columns):
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
            return self._read_until_delimiters()
        elif rule == EscapingRule.CSV:
            return self._read_csv_field()
        elif rule == EscapingRule.JSON:
            return self._read_json_field()
        elif rule == EscapingRule.QUOTED:
            return self._read_quoted_field()
        else:
            raise ValueError(f"Unsupported rule: {rule.name}")

    def _read_until_delimiters(self) -> str:
        """Read characters until we see any **non-empty** delimiter or newline.
        Skips empty delimiters so we don't stop prematurely.
        """
        # Gather all non-empty delimiters *after* the first one
        # (the first delimiter is the one we just matched).
        non_empty = {d for d in self.format.delimiters[1:] if d}
        # We'll also treat a newline as a stopping condition.
        chunks = []
        while True:
            pos = self.input.tell()
            c = self.input.read(1)
            if not c:
                # EOF
                break

            # Put back the char so we can check if it matches any delimiter
            self.input.seek(pos)

            # Check if it matches any known delimiter or newline
            if self._check_string("\n") or self._check_string("\r\n"):
                # We see a newline => end the field
                break

            matched = False
            for d in non_empty:
                if self._check_string(d):
                    matched = True
                    break
            if matched:
                # We see a delimiter => end of field
                break

            # Otherwise consume the character
            c = self.input.read(1)
            chunks.append(c)

        return "".join(chunks).strip()

    def _read_csv_field(self) -> str:
        text = self._read_until_delimiters()
        text = text.strip()
        # If it's quoted CSV, remove the outer quotes and unescape double quotes:
        if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
            inner = text[1:-1].replace('""', '"')
            return inner
        return text

    def _read_json_field(self) -> Any:
        text = self._read_until_delimiters()
        text = text.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {text}") from e

    def _read_quoted_field(self) -> str:
        """Expects an opening quote, read until closing quote.
        No special backslash escapes except double quotes as repeated ""? Adjust as needed.
        """
        # Check we actually have a leading quote
        start = self.input.read(1)
        if start != '"':
            if start:  # revert that char
                self.input.seek(self.input.tell() - 1)
            raise ValueError("Quoted field must start with '\"'")
        chunks = []
        while True:
            c = self.input.read(1)
            if not c:
                raise EOFError("EOF in quoted field")
            if c == '"':
                # Could be end of field or doubled quote
                pos = self.input.tell()
                nxt = self.input.read(1)
                if nxt != '"':
                    # Not a doubled quote => revert
                    self.input.seek(pos)
                    break
                else:
                    # It's a double quote => literal " in the value
                    chunks.append('"')
            else:
                chunks.append(c)
        return "".join(chunks)

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
