from assistant.control_phrases import (
    ControlCommand,
    detect_control_command,
    has_command_content,
    is_affirmative,
    is_name_mentioned,
)


def test_wake_command_detected():
    assert detect_control_command("wake up vaayu") == ControlCommand.WAKE
    assert detect_control_command("Wake up, Vaayu!") == ControlCommand.WAKE
    assert detect_control_command("vaayu wake up") == ControlCommand.WAKE


def test_wake_command_accepts_single_a_spelling():
    # Both "Vayu" and "Vaayu" are legitimate romanizations - STT output
    # could not be reliably validated for which one comes out (see PLAN.md),
    # so both must be accepted.
    assert detect_control_command("wake up vayu") == ControlCommand.WAKE
    assert detect_control_command("Wake up, Vayu!") == ControlCommand.WAKE


def test_sleep_command_detected():
    assert detect_control_command("sleep vaayu") == ControlCommand.SLEEP
    assert detect_control_command("Vaayu, go to sleep.") == ControlCommand.SLEEP


def test_shutdown_command_detected():
    assert detect_control_command("shutdown vaayu") == ControlCommand.SHUTDOWN
    assert detect_control_command("shut down vaayu") == ControlCommand.SHUTDOWN
    assert detect_control_command("Vaayu please shut down") == ControlCommand.SHUTDOWN


def test_no_false_positive_without_name():
    # These contain trigger words but not "vaayu" - must never match, since
    # ordinary conversation shouldn't be able to accidentally control the
    # assistant's running state.
    assert detect_control_command("I need to sleep soon") == ControlCommand.NONE
    assert detect_control_command("please wake up early tomorrow") == ControlCommand.NONE
    assert detect_control_command("can you shut down the laptop") == ControlCommand.NONE


def test_no_false_positive_with_name_but_no_control_word():
    assert detect_control_command("open whatsapp for vaayu") == ControlCommand.NONE
    assert detect_control_command("vaayu what's my battery percentage") == ControlCommand.NONE


def test_empty_text():
    assert detect_control_command("") == ControlCommand.NONE


def test_is_name_mentioned():
    assert is_name_mentioned("vaayu open brave") is True
    assert is_name_mentioned("vayu open brave") is True
    assert is_name_mentioned("open brave") is False
    assert is_name_mentioned("") is False


def test_has_command_content_bare_wake_phrase():
    # A bare wake phrase alone must NOT be treated as having a command
    # attached, or "wake up Vaayu" would try to process itself as a command.
    assert has_command_content("wake up vaayu") is False
    assert has_command_content("vaayu wake up") is False
    assert has_command_content("hey vaayu") is False


def test_has_command_content_combined_wake_and_command():
    # "Vaayu, open brave" (wake + command in one breath) must be detected
    # as having real content beyond just the name/wake words.
    assert has_command_content("vaayu open brave") is True
    assert has_command_content("wake up vaayu and open youtube") is True


def test_is_affirmative_true_for_common_yes_phrasings():
    assert is_affirmative("yes") is True
    assert is_affirmative("yeah go ahead") is True
    assert is_affirmative("yep send it") is True
    assert is_affirmative("sure") is True
    assert is_affirmative("okay do it") is True
    assert is_affirmative("that's correct") is True


def test_is_affirmative_false_for_common_no_phrasings():
    assert is_affirmative("no") is False
    assert is_affirmative("nope") is False
    assert is_affirmative("no don't send it") is False
    assert is_affirmative("wait, cancel that") is False
    assert is_affirmative("") is False


def test_is_affirmative_negative_wins_even_with_incidental_words():
    # "no, that's not right" contains "right"-adjacent phrasing but the
    # explicit "no" must dominate - a decline must never be read as a yes.
    assert is_affirmative("no that's not what I meant") is False


def test_is_affirmative_unrelated_speech_is_not_a_yes():
    # Silence-filler/unrelated transcript must not default to "confirmed" -
    # only recognizably affirmative words should count.
    assert is_affirmative("what time is it") is False
    assert is_affirmative("umm") is False
