"""Phase 0 smoke tests."""

from pathlib import Path


def test_phase0_project_files_exist() -> None:
    project_root = Path(__file__).parents[1]
    assert (project_root / "app.py").is_file()
    assert (project_root / "requirements.txt").is_file()
    assert (project_root / "config" / "settings.py").is_file()
    assert (project_root / "src" / "utils" / "session_state.py").is_file()
