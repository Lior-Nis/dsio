from __future__ import annotations

import subprocess
from pathlib import Path

from dsio.tracking.execution.environment import capture_environment
from dsio.tracking.execution.git import capture_git, working_tree_patch


def test_clean_and_dirty_git_identity(git_repo: Path) -> None:
    clean = capture_git(cwd=git_repo)
    assert clean.sha is not None
    assert clean.dirty is False
    assert clean.code_hash == clean.sha

    (git_repo / "tracked.txt").write_text("modified\n")
    (git_repo / "untracked.txt").write_text("new file\n")
    dirty = capture_git(cwd=git_repo)
    assert dirty.dirty is True
    assert dirty.code_hash is not None
    assert dirty.code_hash.startswith(f"{dirty.sha}-dirty-")
    assert dirty.patch_sha256 is not None


def test_dirty_hash_tracks_file_content(git_repo: Path) -> None:
    (git_repo / "tracked.txt").write_text("first change\n")
    first = capture_git(cwd=git_repo).code_hash
    (git_repo / "tracked.txt").write_text("second change\n")
    assert capture_git(cwd=git_repo).code_hash != first


def test_text_from_untracked_files_is_reconstructible(git_repo: Path) -> None:
    (git_repo / "untracked.txt").write_text("hello from an untracked file\n")
    patch = working_tree_patch(cwd=git_repo).decode()
    assert "hello from an untracked file" in patch
    assert "untracked binary" not in patch


def test_dirty_patch_applies_to_the_recorded_commit(git_repo: Path, tmp_path: Path) -> None:
    source = "def foo():\n    return 1\n\n\n\ndef bar():\n    return 2\n"
    (git_repo / "module.py").write_text(source)
    subprocess.run(["git", "add", "module.py"], cwd=git_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add module"], cwd=git_repo, check=True)
    (git_repo / "module.py").write_text(source.replace("return 1", "return 100"))
    patch = working_tree_patch(cwd=git_repo)

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(git_repo), str(clone)], check=True)
    patch_path = clone / "dirty.patch"
    patch_path.write_bytes(patch)
    applied = subprocess.run(
        ["git", "apply", str(patch_path)],
        cwd=clone,
        capture_output=True,
        text=True,
    )

    assert applied.returncode == 0, applied.stderr
    assert (clone / "module.py").read_text() == source.replace("return 1", "return 100")


def test_missing_git_yields_unknown_identity(tmp_path: Path) -> None:
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    state = capture_git(cwd=plain)
    assert state.sha is None
    assert state.code_hash is None
    assert state.available is False


def test_environment_capture_hashes_the_lockfile() -> None:
    environment = capture_environment(lock_path=Path("uv.lock"))
    assert environment.python
    assert environment.lock_sha256 is not None
