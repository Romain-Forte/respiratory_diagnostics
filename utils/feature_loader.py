from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


def _load_feature_config(config_path: str | Path) -> Mapping[str, object]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature configuration file not found: {path}")

    with path.open(encoding="utf-8") as json_file:
        raw_content = json.load(json_file)

    if not isinstance(raw_content, dict):
        raise ValueError("Feature configuration must be a JSON object mapping names to lists.")
    return raw_content


def _expand_items(items: Sequence[str], available_columns: Optional[Sequence[str]]) -> List[str]:
    expanded_list: List[str] = []
    for item in items:
        if "*" in item:
            if available_columns is None:
                raise ValueError(
                    f"Wildcard '{item}' requires `available_columns` to be provided."
                )
            pattern = "^" + re.escape(item).replace(r"\*", ".*") + "$"
            regex = re.compile(pattern)
            matches = [col for col in available_columns if regex.match(col)]
            if not matches:
                raise ValueError(f"No columns match wildcard '{item}'.")
            expanded_list.extend(matches)
        else:
            expanded_list.append(item)
    return expanded_list


def load_feature_lists(
    config_path: str | Path,
    sections: Sequence[str] | None = None,
    available_columns: Optional[Sequence[str]] = None,
) -> Dict[str, List[str]]:
    """
    Load feature name lists from a JSON configuration file.

    If ``sections`` is provided, only the requested keys will be returned.
    Otherwise all list-based or composite entries in the JSON file are returned.

    Each section can either be a plain list of feature names or a dictionary
    with ``sections`` (list of other section names), optional ``extra`` feature
    names, and optional ``exclude`` names.

    If a feature entry contains a '*' wildcard, all matching column names are
    expanded using the provided ``available_columns`` list, treating '*'
    as the regex token ``.*`` within the column name.
    """
    raw_content = _load_feature_config(config_path)

    def _resolve_section(
        section_name: str,
        stack: Tuple[str, ...],
        cache: Dict[str, List[str]],
    ) -> List[str]:
        if section_name in cache:
            return cache[section_name]
        if section_name in stack:
            cycle = " -> ".join(stack + (section_name,))
            raise ValueError(f"Circular section dependency detected: {cycle}")
        if section_name not in raw_content:
            raise KeyError(f"Section '{section_name}' not found in feature configuration.")

        section_value = raw_content[section_name]

        if isinstance(section_value, list):
            if not all(isinstance(item, str) for item in section_value):
                raise ValueError(f"Section '{section_name}' must contain only strings.")
            columns = _expand_items(section_value, available_columns)
        elif isinstance(section_value, dict) and any(
            key in section_value for key in ("sections", "extra", "exclude")
        ):
            nested_sections = section_value.get("sections") or []
            if not isinstance(nested_sections, list) or not all(
                isinstance(item, str) for item in nested_sections
            ):
                raise ValueError(
                    f"Section '{section_name}' field 'sections' must be a list of strings."
                )
            columns: List[str] = []
            for nested in nested_sections:
                columns.extend(
                    _resolve_section(nested, stack + (section_name,), cache)
                )

            extra_items = section_value.get("extra") or []
            if not isinstance(extra_items, list) or not all(
                isinstance(item, str) for item in extra_items
            ):
                raise ValueError(
                    f"Section '{section_name}' field 'extra' must be a list of strings."
                )
            if extra_items:
                columns.extend(_expand_items(extra_items, available_columns))

            exclude_items = section_value.get("exclude") or []
            if not isinstance(exclude_items, list) or not all(
                isinstance(item, str) for item in exclude_items
            ):
                raise ValueError(
                    f"Section '{section_name}' field 'exclude' must be a list of strings."
                )
            if exclude_items:
                exclude_columns = set(_expand_items(exclude_items, available_columns))
                columns = [col for col in columns if col not in exclude_columns]

            columns = list(dict.fromkeys(columns))
        else:
            raise ValueError(
                f"Section '{section_name}' must be either a list of strings or a "
                "dictionary containing 'sections', 'extra', or 'exclude'."
            )

        cache[section_name] = columns
        return columns

    requested_sections: List[str]
    if sections is None:
        requested_sections = [
            key
            for key, value in raw_content.items()
            if isinstance(value, list)
            or (
                isinstance(value, dict)
                and any(field in value for field in ("sections", "extra", "exclude"))
            )
        ]
    else:
        requested_sections = sections

    feature_lists: Dict[str, List[str]] = {}
    cache: Dict[str, List[str]] = {}
    for section_name in requested_sections:
        feature_lists[section_name] = _resolve_section(section_name, tuple(), cache)

    return feature_lists


def load_columns(
    config_path: str | Path,
    sections: Sequence[str] | None = None,
    available_columns: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Convenience wrapper that returns a single flattened list of columns.

    Args:
        config_path: Path to the JSON file.
        sections: Optional iterable of keys to load from the configuration.
                  When omitted, all sections are concatenated.
    """
    feature_lists = load_feature_lists(config_path, sections, available_columns)
    columns: List[str] = []
    for values in feature_lists.values():
        columns.extend(values)
    return columns


def load_diagnostic_feature_map(
    config_path: str | Path,
    diagnoses: Optional[Sequence[str]] = None,
    available_columns: Optional[Sequence[str]] = None,
) -> Dict[str, List[str]]:
    """
    Load the feature subsets to use per diagnostic.

    The JSON file can contain a ``diagnostic_feature_sets`` entry shaped as:

    {
        "diagnostic_feature_sets": {
            "default": {
                "sections": ["Liste_features_simple"],
                "extra": ["CustomColumn1"],
                "exclude": ["SomeColumn*"]
            },
            "Some Diagnosis": {
                "sections": ["Liste_features_simple", "extra_block"],
                "extra": [],
                "exclude": []
            }
        }
    }

    Each diagnostic can reference shared sections defined elsewhere in the
    configuration, append ``extra`` feature names (including wildcards) and
    optionally ``exclude`` some columns. Each diagnostic entry must specify the
    sections it relies on.
    """
    raw_content = _load_feature_config(config_path)
    diag_sets = raw_content.get("diagnostic_feature_sets", {})
    if not isinstance(diag_sets, dict):
        raise ValueError("`diagnostic_feature_sets` must be a dictionary.")

    if diagnoses is None:
        target_diagnoses = list(diag_sets.keys())
    else:
        target_diagnoses = list(diagnoses)

    result: Dict[str, List[str]] = {}
    for diagnosis in target_diagnoses:
        entry = diag_sets.get(diagnosis)
        if entry is None:
            raise KeyError(
                f"No diagnostic feature set configured for '{diagnosis}'."
            )
        if isinstance(entry, list):
            entry = {"sections": entry, "extra": [], "exclude": []}
        elif not isinstance(entry, dict):
            raise ValueError(
                f"Diagnostic entry for '{diagnosis}' must be a dict or list."
            )

        sections = entry.get("sections")
        if sections is None:
            raise ValueError(
                f"Diagnostic entry for '{diagnosis}' must include a 'sections' list."
            )
        extra_items = entry.get("extra", [])
        exclude_items = entry.get("exclude", [])

        columns: List[str] = []
        section_lists = load_feature_lists(
            config_path, sections=sections, available_columns=available_columns
        )
        for values in section_lists.values():
            columns.extend(values)

        if extra_items:
            columns.extend(_expand_items(extra_items, available_columns))

        if exclude_items:
            exclude_columns = set(_expand_items(exclude_items, available_columns))
            columns = [col for col in columns if col not in exclude_columns]

        result[diagnosis] = list(dict.fromkeys(columns))

    return result
