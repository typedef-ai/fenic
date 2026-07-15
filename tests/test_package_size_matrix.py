import importlib.util
import sys
from pathlib import Path


def _load_package_size_matrix():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "package_size_matrix.py"
    spec = importlib.util.spec_from_file_location("package_size_matrix", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_combo_handles_core_explicit_extras_and_all():
    package_size_matrix = _load_package_size_matrix()

    assert package_size_matrix.parse_combo("core", ["pdf"]) == ("core", [])
    assert package_size_matrix.parse_combo("pdf, cluster", ["cloud", "pdf"]) == (
        "pdf,cluster",
        ["pdf", "cluster"],
    )
    assert package_size_matrix.parse_combo("all", ["cloud", "pdf"]) == (
        "all",
        ["cloud", "pdf"],
    )


def test_pyproject_for_uses_local_fenic_with_selected_extras(tmp_path):
    package_size_matrix = _load_package_size_matrix()

    pyproject = package_size_matrix.pyproject_for(
        tmp_path,
        ["pdf", "cluster"],
        "3.11",
    )

    assert 'requires-python = ">=3.11"' in pyproject
    assert f'"fenic[pdf,cluster] @ {tmp_path.as_uri()}"' in pyproject


def test_write_json_and_csv_include_summary_fields(tmp_path):
    package_size_matrix = _load_package_size_matrix()
    result = package_size_matrix.SizeResult(
        label="pdf",
        extras=["pdf"],
        package_count=2,
        site_packages_bytes=1024,
        venv_bytes=2048,
        top_packages=[
            package_size_matrix.PackageSize(
                name="fenic",
                version="1.2.3",
                bytes=512,
            )
        ],
    )

    json_path = tmp_path / "sizes.json"
    csv_path = tmp_path / "sizes.csv"
    package_size_matrix.write_json(json_path, [result])
    package_size_matrix.write_csv(csv_path, [result])

    assert '"site_packages_bytes": 1024' in json_path.read_text()
    assert "pdf,pdf,2,1024,2048" in csv_path.read_text()
