from pathlib import Path
from typing import Union

from fenic.core.error import ValidationError


def validate_paths_and_return_list_of_strings(paths: Union[str, Path, list[str]]) -> list[str]:
    """Validate paths arg type and return a list of strings."""
    # Validate paths type
    if not isinstance(paths, (str, Path, list)):
        raise ValidationError(
            f"Expected paths to be str, Path, or list, got {type(paths).__name__}"
        )

    # Convert to list if it's a single path
    if isinstance(paths, (str, Path)):
        paths_str_list = [str(paths)]
    else:
        paths_str_list = []
        # Validate each item in the list
        for i, path in enumerate(paths):
            if not isinstance(path, (str, Path)):
                raise ValidationError(
                    f"Expected path at index {i} to be str or Path, got {type(path).__name__}"
                )
            paths_str_list.append(str(path))
    return paths_str_list
