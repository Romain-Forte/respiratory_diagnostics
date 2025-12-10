from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def load_feature_lists(
    config_path: str | Path,
    sections: Sequence[str] | None = None,
) -> Dict[str, List[str]]:
    """
    Load feature name lists from a JSON configuration file.

    If ``sections`` is provided, only the requested keys will be returned.
    Otherwise all the lists contained in the JSON file are returned.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature configuration file not found: {path}")

    with path.open(encoding="utf-8") as json_file:
        raw_content = json.load(json_file)

    if not isinstance(raw_content, dict):
        raise ValueError("Feature configuration must be a JSON object mapping names to lists.")

    requested_sections: Iterable[str]
    if sections is None:
        requested_sections = raw_content.keys()
    else:
        requested_sections = sections

    feature_lists: Dict[str, List[str]] = {}
    for section_name in requested_sections:
        if section_name not in raw_content:
            raise KeyError(f"Section '{section_name}' not found in feature configuration.")
        section_value = raw_content[section_name]
        if not isinstance(section_value, list) or not all(isinstance(item, str) for item in section_value):
            raise ValueError(f"Section '{section_name}' must be a list of strings.")
        feature_lists[section_name] = section_value

    return feature_lists


def load_columns(config_path: str | Path, sections: Sequence[str] | None = None) -> List[str]:
    """
    Convenience wrapper that returns a single flattened list of columns.

    Args:
        config_path: Path to the JSON file.
        sections: Optional iterable of keys to load from the configuration.
                  When omitted, all sections are concatenated.
    """
    feature_lists = load_feature_lists(config_path, sections)
    columns: List[str] = []
    for values in feature_lists.values():
        columns.extend(values)
    return columns
