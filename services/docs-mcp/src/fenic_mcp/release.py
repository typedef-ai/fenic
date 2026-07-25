"""Release-time version validation shared by CI and Modal image construction."""

import importlib.metadata
import re

_VERSION_PATTERN = re.compile(
    r"^(?P<version>0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:[-+][0-9A-Za-z.-]+)?$"
)


def normalize_fenic_version(value: str) -> str:
    """Return a validated package version, accepting an optional ``v`` prefix."""
    version = value.removeprefix("v")
    if not _VERSION_PATTERN.fullmatch(version):
        raise ValueError(f"Invalid Fenic release version: {value!r}")
    return version


def assert_installed_fenic_version(expected: str) -> None:
    """Fail when the service environment is not using the requested release."""
    expected = normalize_fenic_version(expected)
    installed = importlib.metadata.version("fenic")
    if installed != expected:
        raise RuntimeError(
            f"Expected fenic {expected}, but the environment contains {installed}"
        )
