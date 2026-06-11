import json
import subprocess
import sys

SCRIPT = "tools/benchmark_semantic_operator_memory.py"


def test_semantic_operator_memory_harness_emits_machine_readable_case_result():
    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--cases",
            "semantic_join",
            "--rows",
            "2",
            "--label",
            "pytest",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert payload["label"] == "pytest"
    assert payload["measurement"]["rss_source"] == "resource.getrusage(RUSAGE_SELF).ru_maxrss"
    assert payload["measurement"]["polars_allocation_bytes"] is None
    assert "does not expose a process peak allocator counter" in payload["measurement"]["polars_allocation_note"]
    assert [case["case"] for case in payload["cases"]] == ["semantic_join"]

    case = payload["cases"][0]
    assert case["peak_rss_bytes"] > 0
    assert case["elapsed_ms"] >= 0
    assert case["result_rows"] >= 0
    assert case["parameters"]["rows"] == 2
    assert case["parameters"]["network_calls"] == "disabled"


def test_semantic_operator_memory_harness_markdown_keeps_copyable_json_block():
    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--cases",
            "sim_join",
            "--rows",
            "2",
            "--label",
            "markdown-test",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "# Semantic operator memory benchmark" in result.stdout
    assert "```json" in result.stdout
    json_block = result.stdout.split("```json", 1)[1].split("```", 1)[0].strip()
    payload = json.loads(json_block)
    assert payload["cases"][0]["case"] == "sim_join"
    assert payload["cases"][0]["peak_rss_bytes"] > 0
