from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Any


def add_settings_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--settings",
        dest="settings_path",
        default=None,
        help="Load option defaults from a settings file (json or csv).",
    )
    parser.add_argument(
        "--save-settings",
        dest="save_settings_path",
        default=None,
        help="Save current option values to a settings file (json or csv).",
    )


def strip_settings_args(
    argv: Iterable[str],
) -> tuple[list[str], str | None, str | None]:
    cleaned: list[str] = []
    settings_path: str | None = None
    save_path: str | None = None

    it = list(argv)
    i = 0
    while i < len(it):
        arg = it[i]
        if arg == "--settings":
            if i + 1 >= len(it):
                raise SystemExit("--settings requires a path.")
            settings_path = it[i + 1]
            i += 2
            continue
        if arg.startswith("--settings="):
            settings_path = arg.split("=", 1)[1]
            i += 1
            continue
        if arg == "--save-settings":
            if i + 1 >= len(it):
                raise SystemExit("--save-settings requires a path.")
            save_path = it[i + 1]
            i += 2
            continue
        if arg.startswith("--save-settings="):
            save_path = arg.split("=", 1)[1]
            i += 1
            continue

        cleaned.append(arg)
        i += 1

    return cleaned, settings_path, save_path


def detect_command(argv: Iterable[str]) -> str | None:
    for arg in argv:
        if not arg.startswith("-"):
            return arg
    return None


def _parse_csv_value(raw: str) -> Any:
    text = raw.strip()
    if text == "":
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _load_csv(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            key = row[0].strip()
            if not key:
                continue
            if key.lower() in {"key", "name"} and len(row) > 1:
                if row[1].strip().lower() in {"value", "val"}:
                    continue
            if len(row) < 2:
                data[key] = ""
                continue
            data[key] = _parse_csv_value(row[1])
    return data


def _save_csv(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "value"])
        for key in sorted(data):
            writer.writerow([key, json.dumps(data[key], ensure_ascii=True)])


def load_settings(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Settings file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(path)

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        return data
    raise SystemExit(f"Settings file must be a JSON object: {path}")


def save_settings(
    path: Path,
    settings: dict[str, Any],
    *,
    command: str | None = None,
) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        _save_csv(path, settings)
        return

    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = load_settings(path)
        except SystemExit:
            data = {}

    if command:
        if not isinstance(data, dict):
            data = {}
        if data and all(not isinstance(v, dict) for v in data.values()):
            data = {"default": data}
        data[command] = settings
    else:
        data = settings

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")


def select_settings(
    data: dict[str, Any],
    command: str | None,
) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}

    if command and command in data and isinstance(data[command], dict):
        return dict(data[command])

    for key in ("default", "__all__"):
        if key in data and isinstance(data[key], dict):
            return dict(data[key])

    if all(not isinstance(v, dict) for v in data.values()):
        return dict(data)

    return {}


def _iter_actions(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    return [action for action in parser._actions if hasattr(action, "dest")]


def apply_settings_to_parser(
    parser: argparse.ArgumentParser,
    settings: dict[str, Any],
) -> None:
    if not settings:
        return
    for action in _iter_actions(parser):
        if not action.option_strings:
            continue
        if action.dest not in settings:
            continue
        action.default = settings[action.dest]
        if getattr(action, "required", False):
            action.required = False


def collect_option_dests(parser: argparse.ArgumentParser) -> set[str]:
    dests: set[str] = set()
    for action in _iter_actions(parser):
        if not action.option_strings:
            continue
        dests.add(action.dest)
    return dests


def _coerce_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def serialize_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    if exclude is None:
        exclude = set()
    option_dests = collect_option_dests(parser)
    out: dict[str, Any] = {}
    for dest in option_dests:
        if dest in exclude:
            continue
        value = getattr(args, dest, None)
        out[dest] = _coerce_value(value)
    return out


def find_subparser(
    parser: argparse.ArgumentParser,
    command: str | None,
) -> argparse.ArgumentParser | None:
    if not command:
        return None
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices.get(command)
    return None
