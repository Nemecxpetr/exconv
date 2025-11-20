# tests/conftest.py
from __future__ import annotations

import sys
from pathlib import Path
import pytest

# Root of repo: tests/.. = project root (adjust if needed)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Where we expect assets
TEST_ASSETS_DIR = PROJECT_ROOT / "samples" / "input" / "test_assets"

# Names we want to exist
AUDIO_FILES = [
    "audio_long_sines.wav",
    "audio_plucks.wav",
]

IMAGE_FILES = [
    "img_checker.png",
    "img_gradients.png",
    "img_radial.png",
]


def _have_all_assets() -> bool:
    for name in AUDIO_FILES + IMAGE_FILES:
        if not (TEST_ASSETS_DIR / name).exists():
            return False
    return True


def _generate_assets():
    """
    Call the project script to generate assets into TEST_ASSETS_DIR.
    We assume scripts/generate_test_assests.py defines main(out_folder=...).
    """
    try:
        # scripts is a top-level package; make sure project root is on sys.path
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts import generate_test_assests  # NOTE: matches your filename typo
    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            f"Could not import scripts.generate_test_assests to generate test assets: {exc}"
        )

    TEST_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Call your script's main, directing output folder explicitly
    if hasattr(generate_test_assests, "main"):
        generate_test_assests.main(out_folder=str(TEST_ASSETS_DIR))
    else:
        # Fallback: if script only runs via __main__, raise a helpful error
        pytest.skip(
            "generate_test_assests has no main(out_folder=...) function. "
            "Please adapt the script or generate assets manually."
        )


@pytest.fixture(scope="session")
def test_assets_dir() -> Path:
    """
    Ensure test assets exist in samples/input/test_assets.
    - If they exist: just return the path.
    - If not:
      * If interactive (stdin is TTY): ask user y/N whether to generate.
      * If non-interactive (CI): generate automatically.
    """
    if _have_all_assets():
        return TEST_ASSETS_DIR

    # Missing: decide what to do
    interactive = sys.stdin is not None and sys.stdin.isatty()

    if interactive:
        print(
            f"[exconv tests] Synthetic test assets not found in {TEST_ASSETS_DIR}.\n"
            "Generate them now using scripts/generate_test_assests.py? [y/N]: ",
            end="",
            flush=True,
        )
        try:
            resp = input().strip().lower()
        except EOFError:
            resp = ""

        if resp.startswith("y"):
            _generate_assets()
        else:
            pytest.skip("User declined to generate synthetic test assets.")
    else:
        # Non-interactive: auto-generate
        _generate_assets()

    # After generation, sanity-check again
    if not _have_all_assets():
        pytest.skip(
            f"Test assets still missing in {TEST_ASSETS_DIR} "
            "after attempting generation."
        )

    return TEST_ASSETS_DIR
