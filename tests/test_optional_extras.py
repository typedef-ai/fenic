import subprocess
import sys
import textwrap


def _run_with_blocked_modules(blocked_modules: list[str], body: str):
    code = f"""
import importlib.abc
import sys

blocked = {blocked_modules!r}


class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise ModuleNotFoundError(f"No module named {{fullname!r}}", name=fullname)
        return None


sys.meta_path.insert(0, Blocker())

{textwrap.dedent(body)}
"""
    return subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )


def test_fenic_imports_without_pdf_cluster_or_sim_join_extras():
    _run_with_blocked_modules(
        ["fitz", "lancedb", "sklearn"],
        """
import fenic
import fenic._backends.local.physical_plan.join
import fenic._backends.local.physical_plan.transform
import fenic._backends.local.semantic_operators.cluster
import fenic._backends.local.semantic_operators.parse_pdf
import fenic._backends.local.semantic_operators.sim_join
import fenic._backends.local.utils.doc_loader
import fenic._inference.request_utils

assert fenic.Session
""",
    )


def test_optional_dependency_errors_include_extra_install_hint():
    result = _run_with_blocked_modules(
        ["fitz", "lancedb", "sklearn"],
        """
from fenic._optional_dependencies import import_optional_dependency

checks = [
    ("fitz", "pdf", "PDF parsing", "fenic[pdf]"),
    ("lancedb", "sim-join", "semantic similarity joins", "fenic[sim-join]"),
    ("sklearn.cluster", "cluster", "semantic clustering", "fenic[cluster]"),
]

for module_name, extra, feature, expected in checks:
    try:
        import_optional_dependency(module_name, extra=extra, feature=feature)
    except ImportError as exc:
        assert expected in str(exc)
    else:
        raise AssertionError(f"{module_name} unexpectedly imported")
""",
    )
    assert result.returncode == 0
