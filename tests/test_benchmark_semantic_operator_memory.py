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


def test_semantic_operator_memory_harness_can_emit_phase_profiles():
    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--cases",
            "semantic_reduce,semantic_join",
            "--rows",
            "4",
            "--right-rows",
            "3",
            "--groups",
            "2",
            "--label",
            "phase-profile-test",
            "--profile-phases",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    cases = {case["case"]: case for case in payload["cases"]}

    reduce_phases = cases["semantic_reduce"]["phase_profile"]
    assert {phase["name"] for phase in reduce_phases} >= {
        "preprocess_group",
        "build_request_messages",
        "model_completions",
    }
    assert all(phase["rss_after_bytes"] > 0 for phase in reduce_phases)

    join_phases = cases["semantic_join"]["phase_profile"]
    assert {phase["name"] for phase in join_phases} >= {
        "build_join_pairs",
        "predicate_execute",
        "postprocess_join",
    }
    assert all(phase["rss_after_bytes"] > 0 for phase in join_phases)
