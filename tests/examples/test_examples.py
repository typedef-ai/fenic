import importlib.util
from pathlib import Path

import pytest

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
@pytest.mark.parametrize("script_path", get_example_scripts())
def test_example_script(script_path, examples_session_config):
    """Test that each example script's main function runs without errors."""
    module = import_module_from_path(script_path)
    assert hasattr(module, "main"), f"Script {script_path} does not have a main() function"
    module.main(examples_session_config)  # Run the main function
