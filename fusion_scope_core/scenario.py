from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


@dataclass(frozen=True)
class Scenario:
    path: Path
    config: dict[str, Any]
    body: str

    @property
    def scenario_id(self) -> str:
        return str(self.config["id"])

    @property
    def benchmark_kind(self) -> str:
        return str(self.config["benchmark_kind"])

    @property
    def name(self) -> str:
        return str(self.config.get("name", self.scenario_id))


def load_scenario(path: str | Path) -> Scenario:
    """Load a Markdown scenario file with TOML front matter.

    The file format is:

        +++
        id = "..."
        benchmark_kind = "..."
        fusion_scopes = [1, 2, 4]
        +++
        # Human-readable scenario notes

    TOML is used instead of YAML because Python 3.11 includes tomllib.
    """
    p = Path(path).expanduser().resolve()
    text = p.read_text(encoding="utf-8")
    if not text.startswith("+++"):
        raise ValueError(f"Scenario file {p} must start with TOML front matter delimited by +++")
    parts = text.split("+++", 2)
    if len(parts) < 3:
        raise ValueError(f"Scenario file {p} has malformed TOML front matter")
    _, raw_toml, body = parts
    config = tomllib.loads(raw_toml.strip())
    required = ["id", "benchmark_kind", "fusion_scopes", "default_N", "block", "warmup", "repeat", "trials"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Scenario file {p} misses required fields: {missing}")
    return Scenario(path=p, config=config, body=body.lstrip())


def comma_join(values: list[int] | tuple[int, ...]) -> str:
    return ",".join(str(int(x)) for x in values)
