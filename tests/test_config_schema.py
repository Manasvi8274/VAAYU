import pytest

from assistant.config_schema import AppConfig, load_config, set_check_in_time


def test_load_config_raises_clear_error_when_file_missing(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError, match="config.example.yaml"):
        load_config(missing)


def test_load_config_reads_real_yaml_and_applies_defaults(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
llm:
  model: "qwen2.5:3b"
tts:
  rate: 250
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.llm.model == "qwen2.5:3b"
    assert config.tts.rate == 250
    # untouched sections still get their defaults
    assert config.activation.mode == "continuous"
    assert config.stt.whisper_model_size == "base"


def test_load_config_empty_file_uses_all_defaults(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("", encoding="utf-8")

    config = load_config(config_file)

    assert config == AppConfig()


def test_files_config_resolves_user_relative_paths():
    config = AppConfig()
    resolved = config.files.resolved_roots()
    assert len(resolved) == 3
    assert all(p.is_absolute() for p in resolved)
    assert all("~" not in str(p) for p in resolved)


def test_default_self_review_check_in_time_is_ist_9pm():
    # Matches explicit instruction: "the timings will be in IST until I
    # mention any timezone format" and the example "9-10 pm IST".
    config = AppConfig()
    assert config.self_review.timezone == "Asia/Kolkata"
    assert config.self_review.check_in_time == "21:00"


# --- set_check_in_time: targeted text edits, must never destroy comments ---


def test_set_check_in_time_appends_new_section_when_absent(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text('llm:\n  model: "qwen2.5:3b"\n', encoding="utf-8")

    set_check_in_time("21:00", path=config_file)

    config = load_config(config_file)
    assert config.self_review.check_in_time == "21:00"
    assert config.llm.model == "qwen2.5:3b"  # untouched


def test_set_check_in_time_updates_existing_value(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        'self_review:\n  check_in_time: "21:00"\n  timezone: "Asia/Kolkata"\n', encoding="utf-8"
    )

    set_check_in_time("18:30", path=config_file)

    config = load_config(config_file)
    assert config.self_review.check_in_time == "18:30"
    assert config.self_review.timezone == "Asia/Kolkata"  # sibling key untouched


def test_set_check_in_time_preserves_comments_and_unrelated_sections(tmp_path):
    config_file = tmp_path / "config.yaml"
    original = (
        "# a helpful comment\n"
        "llm:\n"
        '  host: "http://localhost:11434"\n'
        "  # another comment explaining the model choice\n"
        '  model: "qwen2.5:3b"\n'
        "\n"
        "tts:\n"
        "  rate: 210   # inline comment\n"
    )
    config_file.write_text(original, encoding="utf-8")

    set_check_in_time("21:00", path=config_file)

    new_text = config_file.read_text(encoding="utf-8")
    assert "# a helpful comment" in new_text
    assert "# another comment explaining the model choice" in new_text
    assert "# inline comment" in new_text
    assert 'model: "qwen2.5:3b"' in new_text
    assert "rate: 210" in new_text


def test_set_check_in_time_can_be_called_repeatedly(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text('llm:\n  model: "qwen2.5:3b"\n', encoding="utf-8")

    set_check_in_time("21:00", path=config_file)
    set_check_in_time("18:30", path=config_file)
    set_check_in_time("09:15", path=config_file)

    config = load_config(config_file)
    assert config.self_review.check_in_time == "09:15"
