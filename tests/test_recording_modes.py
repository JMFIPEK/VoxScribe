"""Tests for voxscribe.recording_modes (stdlib only, real Windows device names)."""

import pytest

from voxscribe import recording_modes as rm

JABRA_HEADSET_MIC = {"index": 15, "name": "Kopfhörermikrofon (2- Jabra Link 390)"}
LAPTOP_MIC_ARRAY = {"index": 14, "name": "Mikrofonarray (Intel® Smart Sound Technologie für digitale Mikrofone)"}
RAZER_MIC = {"index": 3, "name": "Mikrofon (Razer BlackShark V2 Pro 2.4)"}
SPEAKERPHONE_MIC = {"index": 7, "name": "Echo Cancelling Speakerphone (Jabra Speak 510 USB)"}

JABRA_HEADSET_OUT = "Kopfhörer (2- Jabra Link 390) [Loopback]"
LAPTOP_SPEAKERS_OUT = "Lautsprecher (4- Cirrus Logic XU) [Loopback]"
RAZER_OUT = "Lautsprecher (Razer BlackShark V2 Pro 2.4) [Loopback]"
SPEAKERPHONE_OUT = "Echo Cancelling Speakerphone (Jabra Speak 510 USB) [Loopback]"


@pytest.mark.parametrize("mode, source", [
    (rm.MODE_PLAYBACK, "system"), (rm.MODE_CALL, "both"), (rm.MODE_ROOM, "mic"),
])
def test_mode_and_source_map_both_ways(mode, source):
    assert rm.source_for_mode(mode) == source
    assert rm.mode_for_source(source) == mode


def test_headset_with_its_own_mic_is_headset():
    setup, mic = rm.guess_setup([LAPTOP_MIC_ARRAY, JABRA_HEADSET_MIC], JABRA_HEADSET_OUT)
    assert setup == rm.SETUP_HEADSET
    assert mic == JABRA_HEADSET_MIC


def test_headset_whose_output_is_called_lautsprecher_is_still_headset():
    # Windows calls many headsets' output "Lautsprecher"/"Speakers" - only
    # the physical device name in parentheses counts.
    setup, mic = rm.guess_setup([LAPTOP_MIC_ARRAY, RAZER_MIC], RAZER_OUT)
    assert setup == rm.SETUP_HEADSET
    assert mic == RAZER_MIC


def test_laptop_speakers_without_matching_mic_are_loudspeaker_with_mic_array():
    setup, mic = rm.guess_setup([JABRA_HEADSET_MIC, LAPTOP_MIC_ARRAY], LAPTOP_SPEAKERS_OUT)
    assert setup == rm.SETUP_LOUDSPEAKER
    assert mic == LAPTOP_MIC_ARRAY


def test_loudspeaker_without_any_room_mic_keeps_default_input():
    setup, mic = rm.guess_setup([JABRA_HEADSET_MIC], LAPTOP_SPEAKERS_OUT)
    assert setup == rm.SETUP_LOUDSPEAKER
    assert mic is None


def test_speakerphone_is_loudspeaker_despite_shared_device():
    setup, mic = rm.guess_setup([LAPTOP_MIC_ARRAY, SPEAKERPHONE_MIC], SPEAKERPHONE_OUT)
    assert setup == rm.SETUP_LOUDSPEAKER
    assert mic == SPEAKERPHONE_MIC


def test_mic_array_on_same_device_as_speakers_is_loudspeaker():
    mic_array = {"index": 1, "name": "Microphone Array (Realtek(R) Audio)"}
    setup, _mic = rm.guess_setup([mic_array], "Speakers (Realtek(R) Audio) [Loopback]")
    assert setup == rm.SETUP_LOUDSPEAKER


def test_no_playback_device_is_loudspeaker():
    assert rm.guess_setup([JABRA_HEADSET_MIC], None) == (rm.SETUP_LOUDSPEAKER, None)


def test_mic_matches_playback():
    assert rm.mic_matches_playback(JABRA_HEADSET_MIC["name"], JABRA_HEADSET_OUT)
    assert not rm.mic_matches_playback(LAPTOP_MIC_ARRAY["name"], JABRA_HEADSET_OUT)
    assert not rm.mic_matches_playback(JABRA_HEADSET_MIC["name"], None)


def test_mic_matches_playback_with_mme_truncated_name():
    assert rm.mic_matches_playback("Mikrofon (Razer BlackShark V2 P", RAZER_OUT)
