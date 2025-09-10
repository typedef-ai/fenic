from google.genai.types import Content, ContentUnion, Part

from fenic._inference.types import LMRequestMessages


def convert_messages(messages: LMRequestMessages) -> list[ContentUnion]:
        """Convert Fenic LMRequestMessages → list of google-genai `Content` objects.

        Converts Fenic message format to Google's Content format, including
        few-shot examples and the final user prompt.

        Args:
            messages: Fenic message format

        Returns:
            List of Google Content objects
        """
        contents: list[ContentUnion] = []
        # few-shot examples
        for example in messages.examples:
            contents.append(
                Content(
                    role="user", parts=[Part(text=example.user)]
                )
            )
            contents.append(
                Content(
                    role="model", parts=[Part(text=example.assistant)]
                )
            )

        # final user prompt
        contents.append(
            Content(
                role="user", parts=[Part(text=messages.user)]
            )
        )
        return contents