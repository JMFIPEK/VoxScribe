"""Recording names (#51): `YYMMDD_HHMM_<Recording title>.wav`, the title
cleaned up or the Recording mode's default, and never an existing file's name."""

from datetime import datetime

import pytest

from voxscribe import recording_names as rn

STARTED = datetime(2026, 10, 27, 14, 30, 59)


def test_a_title_ends_the_stem_after_the_start_minute():
    assert rn.recording_stem("ASAM WG", "call", STARTED) == "261027_1430_ASAM_WG"


@pytest.mark.parametrize("mode, default", [
    ("call", "Meeting"), ("playback", "Playback"), ("room", "Room"),
])
def test_without_a_title_the_recording_is_named_after_its_mode(mode, default):
    assert rn.recording_stem(None, mode, STARTED) == f"261027_1430_{default}"
    assert rn.recording_stem("", mode, STARTED) == f"261027_1430_{default}"
    assert rn.recording_stem("   ", mode, STARTED) == f"261027_1430_{default}"


def test_characters_windows_forbids_are_dropped_umlauts_and_case_kept():
    assert rn.clean_title('Prüfung: "Gruppe" A/B <Ü>?*|\\ Tag\t2\x01', "call") == "Prüfung_Gruppe_AB_Ü_Tag_2"
    assert rn.clean_title("ASAM : WG", "call") == "ASAM_WG"


def test_a_long_title_is_cut_to_60_characters_without_a_trailing_underscore():
    title = "a" * 59 + " bcd"
    assert rn.clean_title(title, "call") == "a" * 59
    assert rn.clean_title("x" * 80, "call") == "x" * 60


def test_a_title_does_not_end_in_a_dot():
    assert rn.clean_title("Notes...", "room") == "Notes"


@pytest.mark.parametrize("title", ['???', '<>:"', "...", " _ "])
def test_a_title_with_nothing_left_after_cleanup_is_the_mode_default(title):
    assert rn.clean_title(title, "playback") == "Playback"


def test_the_recording_path_is_the_stem_as_wav_in_the_directory(tmp_path):
    assert rn.recording_path(tmp_path, "ASAM WG", "call", STARTED) == str(tmp_path / "261027_1430_ASAM_WG.wav")


@pytest.mark.parametrize("existing", ["261027_1430_Meeting.wav", "261027_1430_Meeting.tracks.flac"])
def test_a_second_recording_in_the_same_minute_gets_2(tmp_path, existing):
    (tmp_path / existing).write_bytes(b"")
    assert rn.recording_path(tmp_path, None, "call", STARTED) == str(tmp_path / "261027_1430_Meeting_2.wav")


def test_the_next_free_number_is_taken(tmp_path):
    for name in ("261027_1430_Room.wav", "261027_1430_Room_2.wav", "261027_1430_Room_3.tracks.flac"):
        (tmp_path / name).write_bytes(b"")
    assert rn.recording_path(tmp_path, "", "room", STARTED) == str(tmp_path / "261027_1430_Room_4.wav")
