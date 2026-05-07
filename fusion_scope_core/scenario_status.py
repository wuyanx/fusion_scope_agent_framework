from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .scenario import Scenario


METADATA_FILENAME = ".fusion_scope_metadata.json"
METADATA_VERSION = 1

REQUIRED_GENERATED_PATHS = [
    "README.md",
    "run_benchmark.py",
    "plot_results.py",
    "scenario.md",
    "fusion_benchmark/__init__.py",
    "fusion_benchmark/device.py",
    "fusion_benchmark/kernels.py",
    "fusion_benchmark/torch_reference.py",
    "results",
]


@dataclass(frozen=True)
class ScenarioStatus:
    state: str
    message: str
    scenario_id: str
    benchmark_kind: str
    generated_dir: Path
    generator_available: bool
    generated_exists: bool
    metadata_exists: bool
    needs_generator: bool
    needs_materialize: bool
    scenario_sha256: str
    recorded_sha256: str | None
    missing_generated_paths: list[str]


def scenario_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metadata_path(generated_dir: Path) -> Path:
    return generated_dir / METADATA_FILENAME


def build_generated_metadata(scenario: Scenario) -> dict[str, Any]:
    return {
        "metadata_version": METADATA_VERSION,
        "scenario_id": scenario.scenario_id,
        "scenario_name": scenario.name,
        "benchmark_kind": scenario.benchmark_kind,
        "scenario_path": str(scenario.path),
        "scenario_sha256": scenario_sha256(scenario.path),
        "config": scenario.config,
    }


def write_generated_metadata(generated_dir: Path, scenario: Scenario) -> None:
    path = metadata_path(generated_dir)
    path.write_text(
        json.dumps(build_generated_metadata(scenario), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_generated_metadata(generated_dir: Path) -> dict[str, Any] | None:
    path = metadata_path(generated_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find_missing_generated_paths(generated_dir: Path) -> list[str]:
    missing = []
    for relative in REQUIRED_GENERATED_PATHS:
        if not (generated_dir / relative).exists():
            missing.append(relative)
    return missing


def classify_scenario(
    scenario: Scenario,
    generated_dir: Path,
    registered_benchmark_kinds: Iterable[str],
) -> ScenarioStatus:
    registered = set(registered_benchmark_kinds)
    current_sha = scenario_sha256(scenario.path)
    generator_available = scenario.benchmark_kind in registered
    generated_exists = generated_dir.exists()
    metadata = read_generated_metadata(generated_dir) if generated_exists else None
    metadata_exists = metadata is not None
    recorded_sha = None if metadata is None else str(metadata.get("scenario_sha256", ""))
    missing_paths = find_missing_generated_paths(generated_dir) if generated_exists else []

    if not generator_available:
        return ScenarioStatus(
            state="new_benchmark_kind",
            message=(
                f"Scenario {scenario.scenario_id} uses new benchmark_kind={scenario.benchmark_kind}. "
                "A generator and registry entry must be created before running it."
            ),
            scenario_id=scenario.scenario_id,
            benchmark_kind=scenario.benchmark_kind,
            generated_dir=generated_dir,
            generator_available=False,
            generated_exists=generated_exists,
            metadata_exists=metadata_exists,
            needs_generator=True,
            needs_materialize=True,
            scenario_sha256=current_sha,
            recorded_sha256=recorded_sha,
            missing_generated_paths=missing_paths,
        )

    if not generated_exists:
        state = "new_scenario_existing_benchmark"
        message = (
            f"Scenario {scenario.scenario_id} has a supported benchmark_kind={scenario.benchmark_kind} "
            "but no generated benchmark yet. Scripts will be materialized before running."
        )
        needs_materialize = True
    elif missing_paths:
        state = "existing_incomplete"
        message = (
            f"Scenario {scenario.scenario_id} has an incomplete generated benchmark. "
            "Scripts will be regenerated before running."
        )
        needs_materialize = True
    elif not metadata_exists:
        state = "existing_without_metadata"
        message = (
            f"Scenario {scenario.scenario_id} has generated files but no metadata. "
            "Scripts will be regenerated once to record the scenario fingerprint."
        )
        needs_materialize = True
    elif recorded_sha != current_sha:
        state = "existing_modified"
        message = (
            f"Scenario {scenario.scenario_id} was modified since the generated benchmark was created. "
            "Scripts will be regenerated before running."
        )
        needs_materialize = True
    else:
        state = "existing_unchanged"
        message = (
            f"Scenario {scenario.scenario_id} already has an up-to-date generated benchmark. "
            "The existing scripts can be run directly."
        )
        needs_materialize = False

    return ScenarioStatus(
        state=state,
        message=message,
        scenario_id=scenario.scenario_id,
        benchmark_kind=scenario.benchmark_kind,
        generated_dir=generated_dir,
        generator_available=True,
        generated_exists=generated_exists,
        metadata_exists=metadata_exists,
        needs_generator=False,
        needs_materialize=needs_materialize,
        scenario_sha256=current_sha,
        recorded_sha256=recorded_sha,
        missing_generated_paths=missing_paths,
    )
