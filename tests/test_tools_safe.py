import pytest

from assistant.tools import files


@pytest.fixture
def safe_root(tmp_path, monkeypatch):
    """Point the files tool at an isolated temp directory instead of the
    user's real Desktop/Documents/Downloads for the duration of the test."""
    monkeypatch.setattr(files, "_SAFE_ROOTS", [tmp_path])
    return tmp_path


def test_read_file_rejects_path_outside_safe_roots(safe_root, tmp_path_factory):
    outside_dir = tmp_path_factory.mktemp("outside")
    outside_file = outside_dir / "secret.txt"
    outside_file.write_text("nope")

    with pytest.raises(PermissionError):
        files.read_file(str(outside_file))


def test_read_file_rejects_disallowed_extension(safe_root):
    exe_path = safe_root / "program.exe"
    with pytest.raises(ValueError):
        files.read_file(str(exe_path))


def test_read_file_missing_file_raises(safe_root):
    missing = safe_root / "missing.txt"
    with pytest.raises(FileNotFoundError):
        files.read_file(str(missing))


def test_read_file_reads_allowed_file(safe_root):
    target = safe_root / "note.txt"
    target.write_text("hello world")
    assert files.read_file(str(target)) == "hello world"


def test_read_file_resolves_bare_filename_against_safe_root(safe_root):
    # Found live: a bare filename with no directory ("note.txt") used to
    # resolve against the process's own working directory instead of any
    # safe root, so a genuinely existing file was reported as not found.
    target = safe_root / "note.txt"
    target.write_text("hello world")
    assert files.read_file("note.txt") == "hello world"


def test_read_file_bare_filename_still_missing_raises_clearly(safe_root):
    # A bare filename that genuinely doesn't exist in any safe root must
    # still raise, not silently succeed on some unrelated path.
    with pytest.raises(FileNotFoundError):
        files.read_file("does_not_exist_anywhere.txt")


def test_read_file_truncates_large_file(safe_root):
    target = safe_root / "big.txt"
    target.write_text("x" * (files._MAX_READ_BYTES + 1000))
    result = files.read_file(str(target))
    assert result.endswith("...[truncated]")
    assert len(result) < files._MAX_READ_BYTES + 100


def test_search_files_finds_matching_filenames(safe_root):
    (safe_root / "report_final.txt").write_text("data")
    (safe_root / "unrelated.txt").write_text("data")

    results = files.search_files("report")
    assert any("report_final.txt" in r for r in results)
    assert not any("unrelated.txt" in r for r in results)


def test_search_files_respects_max_results(safe_root):
    for i in range(files._MAX_RESULTS + 5):
        (safe_root / f"match_{i}.txt").write_text("data")

    results = files.search_files("match")
    assert len(results) == files._MAX_RESULTS
