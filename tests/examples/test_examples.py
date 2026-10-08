import importlib.util
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

import fenic as fc

# Get the examples directory path
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"

def import_module_from_path(module_path):
    """Helper function to import a module from a file path."""
    spec = importlib.util.spec_from_file_location("module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_example_scripts():
    """Get all Python scripts from example directories."""
    scripts = []
    for example_dir in EXAMPLES_DIR.iterdir():
        if example_dir.is_dir():
            for file in example_dir.glob("*.py"):
                if file.name != "__init__.py":
                    scripts.append(file)
    return scripts

# This smoke test runs each script we provide as examples to ensure they run without errors after changes.
# Every example demonstrates a live semantic operation, so this needs a real provider key.
@pytest.mark.requires_provider_key
@pytest.mark.parametrize("script_path", get_example_scripts())
def test_example_script(script_path, examples_session_config):
    """Test that each example script's main function runs without errors."""
    if script_path.parent.name == "typed_judgments":
        if not os.environ.get("TYPESAFE_API_KEY"):
            pytest.skip("TYPESAFE_API_KEY is required for the TypeSafe examples")
        pytest.importorskip("typesafe_sdk")
        examples_session_config = examples_session_config.model_copy(
            update={
                "semantic": fc.SemanticConfig(
                    language_models={
                        "decisions": fc.TypeSafeLanguageModel(
                            model_name="jev-1.13.0", rpm=60, tpm=64_000
                        ),
                    }
                ),
            }
        )
    module = import_module_from_path(script_path)
    assert hasattr(module, "main"), f"Script {script_path} does not have a main() function"
    module.main(examples_session_config)  # Run the main function


@pytest.mark.parametrize(
    "script_path",
    [path for path in get_example_scripts() if path.parent.name == "typed_judgments"],
)
def test_typesafe_example_skips_without_key(script_path, tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    config = fc.SessionConfig(app_name="example_without_key", db_path=tmp_path)
    with pytest.raises(pytest.skip.Exception, match="TYPESAFE_API_KEY"):
        test_example_script(script_path, config)


@pytest.mark.parametrize(
    "script_path",
    [path for path in get_example_scripts() if path.parent.name == "typed_judgments"],
)
def test_typesafe_example_accepts_isolated_config(script_path, tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    create_session = Mock(side_effect=RuntimeError("example started"))
    monkeypatch.setattr(fc.Session, "get_or_create", create_session)
    config = fc.SessionConfig(
        app_name="isolated_example",
        db_path=tmp_path,
        semantic=fc.SemanticConfig(
            language_models={
                "default": fc.OpenAILanguageModel(
                    model_name="gpt-4.1-nano", rpm=60, tpm=64_000
                ),
            }
        ),
    )
    with pytest.raises(RuntimeError, match="example started"):
        test_example_script(script_path, config)
    supplied = create_session.call_args.args[0]
    assert supplied.app_name == config.app_name
    assert supplied.db_path == config.db_path
    assert isinstance(
        supplied.semantic.language_models["decisions"], fc.TypeSafeLanguageModel
    )
    assert isinstance(config.semantic.language_models["default"], fc.OpenAILanguageModel)
