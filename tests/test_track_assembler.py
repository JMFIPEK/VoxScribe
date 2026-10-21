"""Characterization tests for voxscribe.track_assembler.

They pin what a Call recording produced before the track assembler existed
(the after-recording code of recorder._record_both()/_record_both_macos()):
the golden hashes below were computed with that code on the same inputs.
"""

import hashlib

import numpy as np
import soundfile as sf

from voxscribe import tracks
from voxscribe.track_assembler import TrackAssembler


def _chunks(seed, rate, channels, seconds, frames_per_chunk):
    """Random int16 PCM as the recorder receives it: a list of byte chunks,
    interleaved if stereo."""
    rng = np.random.default_rng(seed)
    pcm = rng.integers(-12000, 12000, size=(int(rate * seconds), channels), dtype=np.int16)
    data = pcm.tobytes()
    step = frames_per_chunk * channels * 2
    return [data[i:i + step] for i in range(0, len(data), step)]


def _record(assembler, mic_chunks, playback_chunks, mic_rate, mic_channels,
            playback_rate, playback_channels):
    """Feeds both sources' chunks in the order they were captured."""
    def timed(source, chunks, rate, channels):
        events, frames = [], 0
        for chunk in chunks:
            frames += len(chunk) // (2 * channels)
            events.append((frames / rate, source, chunk))
        return events

    for _, source, chunk in sorted(timed(tracks.MIC, mic_chunks, mic_rate, mic_channels)
                                   + timed(tracks.SYSTEM, playback_chunks, playback_rate,
                                           playback_channels), key=lambda e: e[0]):
        assembler.add_chunk(source, chunk)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sidecar(audio_path):
    data, sr = sf.read(tracks.tracks_path_for(audio_path), dtype="int16", always_2d=True)
    assert sr == 16000
    return np.ascontiguousarray(data)


def _wav_samples(audio_path):
    data, sr = sf.read(audio_path, dtype="int16")
    assert sr == 16000
    return data


def test_windows_like_call_matches_todays_files(tmp_path):
    # 48kHz stereo microphone, 44.1kHz stereo playback that delivered only
    # half as long - both resampled to 16kHz, playback padded with silence.
    out = str(tmp_path / "call.wav")
    assembler = TrackAssembler(mic_rate=48000, mic_channels=2,
                               playback_rate=44100, playback_channels=2)
    _record(assembler, _chunks(1, 48000, 2, 1.0, 3072), _chunks(2, 44100, 2, 0.5, 2822),
            48000, 2, 44100, 2)

    mix = assembler.save(out, setup="headset")

    assert mix.dtype == np.int16 and len(mix) == 16000
    with open(out, "rb") as f:
        assert _sha(f.read()) == "bcce6ff3db526dfd356d4856dc006a5c1b85373501711b8ac7fa30d9829b0c94"
    sidecar = _sidecar(out)
    assert sidecar.shape == (16000, 2)
    assert not sidecar[8000:, 1].any()  # shorter playback track padded with silence
    assert _sha(sidecar.tobytes()) == "a6d93b3f4b9c6620a0f816720df1fbe50fd1043b0e9b2bb5229d6b7f2f280cb4"
    assert tracks.load_setup(out) == "headset"


def test_macos_like_call_matches_todays_files(tmp_path):
    # 44.1kHz mono microphone shorter than the 16kHz mono playback
    # (SystemAudioCapture's fixed format), no Setup given.
    out = str(tmp_path / "call.wav")
    assembler = TrackAssembler(mic_rate=44100, mic_channels=1,
                               playback_rate=16000, playback_channels=1)
    _record(assembler, _chunks(3, 44100, 1, 0.5, 1024), _chunks(4, 16000, 1, 1.0, 1024),
            44100, 1, 16000, 1)

    mix = assembler.save(out)

    assert len(mix) == 16000
    with open(out, "rb") as f:
        assert _sha(f.read()) == "1df878075ad30de5a6e6c12de93a1b2a0c2ad6dc6d1aa81e214e543b4b481bda"
    sidecar = _sidecar(out)
    assert not sidecar[8000:, 0].any()  # shorter microphone track padded
    assert _sha(sidecar.tobytes()) == "55d057019250e4b6f862867d41f18f9d441c4e573bfd061d5bb0dc4d446a97d4"
    assert tracks.load_setup(out) is None


def test_mix_is_both_tracks_at_065_padded_to_the_longer_one(tmp_path):
    out = str(tmp_path / "call.wav")
    assembler = TrackAssembler(mic_rate=16000, mic_channels=1,
                               playback_rate=16000, playback_channels=1)
    assembler.add_chunk(tracks.MIC, np.array([1000, -1000], dtype=np.int16).tobytes())
    assembler.add_chunk(tracks.MIC, np.array([500], dtype=np.int16).tobytes())
    assembler.add_chunk(tracks.SYSTEM, np.array([2000], dtype=np.int16).tobytes())

    mix = assembler.save(out, setup="loudspeaker")

    assert mix.tolist() == [1950, -650, 325]
    assert _wav_samples(out).tolist() == [1950, -650, 325]
    assert _sidecar(out).tolist() == [[1000, 2000], [-1000, 0], [500, 0]]
    assert tracks.load_setup(out) == "loudspeaker"


def test_mix_peak_is_scaled_down_to_int16_range(tmp_path):
    out = str(tmp_path / "call.wav")
    assembler = TrackAssembler(mic_rate=16000, mic_channels=1,
                               playback_rate=16000, playback_channels=1)
    assembler.add_chunk(tracks.MIC, np.array([30000, 100], dtype=np.int16).tobytes())
    assembler.add_chunk(tracks.SYSTEM, np.array([30000, -100], dtype=np.int16).tobytes())

    assert assembler.save(out).tolist() == [32767, 0]


def test_stereo_chunks_are_downmixed_to_the_channel_mean(tmp_path):
    out = str(tmp_path / "call.wav")
    assembler = TrackAssembler(mic_rate=16000, mic_channels=2,
                               playback_rate=16000, playback_channels=1)
    assembler.add_chunk(tracks.MIC, np.array([100, 300, -50, -150], dtype=np.int16).tobytes())
    assembler.add_chunk(tracks.SYSTEM, np.array([0, 0], dtype=np.int16).tobytes())

    assembler.save(out)

    assert _sidecar(out)[:, 0].tolist() == [200, -100]


def test_has_audio_reports_which_sources_delivered_chunks():
    assembler = TrackAssembler(mic_rate=48000, mic_channels=1,
                               playback_rate=48000, playback_channels=2)
    assembler.add_chunk(tracks.MIC, bytes(4))

    assert assembler.has_audio(tracks.MIC)
    assert not assembler.has_audio(tracks.SYSTEM)


# --- Keeping Call tracks in sync (#16) ------------------------------------
#
# The events below are fed in the order a real Call delivers them: each
# chunk at the moment its last sample was captured. Playback that "isn't
# delivered" for a stretch models WASAPI loopback, which sends no packets
# while nothing plays.

CLICK = 20000


def _clicks_pcm(rate, channels, t0, t1, clicks):
    """int16 PCM from t0 to t1 (seconds): silence with a 2 ms click at each
    time in clicks."""
    n = int(round((t1 - t0) * rate))
    pcm = np.zeros((n, channels), dtype=np.int16)
    for t in clicks:
        start = int(round((t - t0) * rate))
        if 0 <= start < n:
            pcm[start:start + int(0.002 * rate)] = CLICK
    return pcm


def _events(source, rate, channels, spans, chunk_seconds, clicks):
    """(capture time, source, chunk) events for the stretches in spans
    that the source actually delivers."""
    events = []
    frames = int(round(chunk_seconds * rate))
    for t0, t1 in spans:
        pcm = _clicks_pcm(rate, channels, t0, t1, clicks)
        for i in range(0, len(pcm), frames):
            chunk = pcm[i:i + frames]
            events.append((t0 + (i + len(chunk)) / rate, source, chunk.tobytes()))
    return events


def _feed(assembler, *event_lists):
    """Feeds events in capture order; an int instead of a chunk is a report
    that that many frames were dropped."""
    for _, source, data in sorted((e for lst in event_lists for e in lst), key=lambda e: e[0]):
        if isinstance(data, int):
            assembler.add_dropped(source, data)
        else:
            assembler.add_chunk(source, data)


def _click_positions(audio_path):
    """Sample positions (16 kHz) where each track's clicks start."""
    sidecar = _sidecar(audio_path).astype(np.int32)
    positions = []
    for ch in (0, 1):
        loud = np.flatnonzero(np.abs(sidecar[:, ch]) > CLICK // 4)
        starts = loud[np.insert(np.diff(loud) > 160, 0, True)] if len(loud) else loud
        positions.append(starts.tolist())
    return positions


def _assert_aligned(audio_path, expected_times):
    mic, playback = _click_positions(audio_path)
    expected = [int(t * 16000) for t in expected_times]
    assert len(mic) == len(playback) == len(expected), (mic, playback)
    for m, p, e in zip(mic, playback, expected):
        assert abs(m - e) <= 160 and abs(p - e) <= 160, (m, p, e)  # within 10 ms


def test_playback_after_a_long_silence_stays_at_the_microphones_moment(tmp_path):
    out = str(tmp_path / "call.wav")
    clicks = [1.0, 8.0, 9.5]
    assembler = TrackAssembler(mic_rate=16000, mic_channels=1,
                               playback_rate=48000, playback_channels=2)
    mic = _events(tracks.MIC, 16000, 1, [(0, 10)], 0.01, clicks)
    playback = _events(tracks.SYSTEM, 48000, 2, [(0, 2), (7, 10)], 0.064, clicks)

    _feed(assembler, mic, playback)
    assembler.save(out)

    _assert_aligned(out, clicks)


def test_playback_after_several_short_silences_stays_aligned(tmp_path):
    out = str(tmp_path / "call.wav")
    clicks = [0.5, 2.0, 3.5, 5.0, 6.5]
    assembler = TrackAssembler(mic_rate=48000, mic_channels=1,
                               playback_rate=48000, playback_channels=2)
    mic = _events(tracks.MIC, 48000, 1, [(0, 7)], 0.01, clicks)
    playback = _events(tracks.SYSTEM, 48000, 2,
                       [(0, 1), (1.5, 2.5), (3, 4), (4.5, 5.5), (6, 7)], 0.064, clicks)

    _feed(assembler, mic, playback)
    assembler.save(out)

    _assert_aligned(out, clicks)


def test_late_starting_playback_lands_at_the_microphones_moment(tmp_path):
    # WASAPI: the loopback stream is open from the start, but nothing plays
    # for the first 3 s, so its first packet arrives only then.
    out = str(tmp_path / "call.wav")
    clicks = [3.5, 5.0]
    assembler = TrackAssembler(mic_rate=16000, mic_channels=1,
                               playback_rate=48000, playback_channels=2)
    mic = _events(tracks.MIC, 16000, 1, [(0, 6)], 0.01, clicks)
    playback = _events(tracks.SYSTEM, 48000, 2, [(3, 6)], 0.064, clicks)

    _feed(assembler, mic, playback)
    assembler.save(out)

    _assert_aligned(out, clicks)


def test_different_native_rates_and_stereo_stay_aligned(tmp_path):
    out = str(tmp_path / "call.wav")
    clicks = [0.7, 4.2, 7.9]
    assembler = TrackAssembler(mic_rate=48000, mic_channels=2,
                               playback_rate=44100, playback_channels=2)
    mic = _events(tracks.MIC, 48000, 2, [(0, 8.5)], 0.01, clicks)
    playback = _events(tracks.SYSTEM, 44100, 2, [(0, 1.5), (4, 8.5)], 0.064, clicks)

    _feed(assembler, mic, playback)
    assembler.save(out)

    _assert_aligned(out, clicks)


def test_microphone_lead_before_the_first_playback_sample_is_dropped(tmp_path):
    # macOS: the microphone runs while the system-audio helper launches; the
    # helper streams continuously from its first sample on (16 kHz mono).
    out = str(tmp_path / "call.wav")
    assembler = TrackAssembler(mic_rate=48000, mic_channels=1,
                               playback_rate=16000, playback_channels=1,
                               playback_is_continuous=True)
    mic = _events(tracks.MIC, 48000, 1, [(0, 3)], 0.01, [1.0, 2.5])
    playback = _events(tracks.SYSTEM, 16000, 1, [(0.4, 3)], 0.064, [1.0, 2.5])

    _feed(assembler, mic, playback)
    mix = assembler.save(out)

    _assert_aligned(out, [0.6, 2.1])  # both tracks start at the helper's first sample
    mix_loud = np.flatnonzero(np.abs(mix.astype(np.int32)) > CLICK // 4)
    assert abs(int(mix_loud[0]) - int(0.6 * 16000)) <= 160


def test_a_dropped_microphone_buffer_leaves_a_gap_and_later_audio_stays_aligned(tmp_path, capsys):
    out = str(tmp_path / "call.wav")
    clicks = [1.0, 4.0]
    assembler = TrackAssembler(mic_rate=48000, mic_channels=1,
                               playback_rate=48000, playback_channels=2)
    mic = _events(tracks.MIC, 48000, 1, [(0, 2.0), (2.1, 5)], 0.01, clicks)
    mic.append((2.1, tracks.MIC, 4800))  # 100 ms lost
    playback = _events(tracks.SYSTEM, 48000, 2, [(0, 5)], 0.064, clicks)

    _feed(assembler, mic, playback)
    assembler.save(out)

    _assert_aligned(out, clicks)
    assert not _sidecar(out)[int(2.0 * 16000) + 16:int(2.1 * 16000) - 16, 0].any()
    assert "Microphone dropped 100 ms" in capsys.readouterr().out


def test_a_dropped_playback_buffer_leaves_a_gap_and_later_audio_stays_aligned(tmp_path, capsys):
    out = str(tmp_path / "call.wav")
    clicks = [1.0, 4.0]
    assembler = TrackAssembler(mic_rate=16000, mic_channels=1,
                               playback_rate=44100, playback_channels=2)
    mic = _events(tracks.MIC, 16000, 1, [(0, 5)], 0.01, clicks)
    playback = _events(tracks.SYSTEM, 44100, 2, [(0, 2.0), (2.1, 5)], 0.064, clicks)
    playback.append((2.1, tracks.SYSTEM, 4410))  # 100 ms lost

    _feed(assembler, mic, playback)
    assembler.save(out)

    _assert_aligned(out, clicks)
    assert "Playback audio dropped 100 ms" in capsys.readouterr().out


def test_playback_that_never_delivered_becomes_a_silent_track(tmp_path):
    # WASAPI: nothing played during the whole Call.
    out = str(tmp_path / "call.wav")
    assembler = TrackAssembler(mic_rate=16000, mic_channels=1,
                               playback_rate=48000, playback_channels=2)
    _feed(assembler, _events(tracks.MIC, 16000, 1, [(0, 2)], 0.01, [1.0]))

    mix = assembler.save(out)

    assert len(mix) == 32000
    sidecar = _sidecar(out)
    assert sidecar.shape == (32000, 2) and not sidecar[:, 1].any()


def test_playback_after_silences_of_a_tenth_of_a_second_stays_aligned(tmp_path):
    out = str(tmp_path / "call.wav")
    clicks = [0.5, 1.2, 1.9, 2.6]
    assembler = TrackAssembler(mic_rate=16000, mic_channels=1,
                               playback_rate=48000, playback_channels=2)
    mic = _events(tracks.MIC, 16000, 1, [(0, 3)], 0.01, clicks)
    playback = _events(tracks.SYSTEM, 48000, 2,
                       [(0, 1.0), (1.1, 1.7), (1.8, 2.4), (2.5, 3)], 0.064, clicks)

    _feed(assembler, mic, playback)
    assembler.save(out)

    _assert_aligned(out, clicks)


def test_a_late_read_of_buffered_playback_inserts_no_silence(tmp_path):
    # The Playback reader stalls from 1.5 s to 2.0 s, then reads the backlog
    # in one go; the driver reports what is still buffered behind each chunk.
    out = str(tmp_path / "call.wav")
    clicks = [1.0, 2.5]
    assembler = TrackAssembler(mic_rate=16000, mic_channels=1,
                               playback_rate=16000, playback_channels=1)
    mic = _events(tracks.MIC, 16000, 1, [(0, 3)], 0.01, clicks)
    playback = _events(tracks.SYSTEM, 16000, 1, [(0, 3)], 0.064, clicks)
    stalled = [chunk for t, _, chunk in playback if 1.5 < t <= 2.0]
    on_time = [e for e in playback if not 1.5 < e[0] <= 2.0]

    _feed(assembler, [e for e in mic if e[0] <= 2.0], [e for e in on_time if e[0] <= 2.0])
    for i, chunk in enumerate(stalled):
        backlog = sum(len(c) // 2 for c in stalled[i + 1:])
        assembler.add_chunk(tracks.SYSTEM, chunk, buffered_frames=backlog)
    _feed(assembler, [e for e in mic if e[0] > 2.0], [e for e in on_time if e[0] > 2.0])
    assembler.save(out)

    _assert_aligned(out, clicks)


def test_a_continuous_playback_source_is_never_padded(tmp_path):
    # macOS: the helper streams without gaps; a slow pipe read is not one.
    out = str(tmp_path / "call.wav")
    assembler = TrackAssembler(mic_rate=16000, mic_channels=1,
                               playback_rate=16000, playback_channels=1,
                               playback_is_continuous=True)
    mic = _events(tracks.MIC, 16000, 1, [(0, 3)], 0.01, [1.0, 2.5])
    playback = _events(tracks.SYSTEM, 16000, 1, [(0, 3)], 0.064, [1.0, 2.5])
    late = [(t + 0.4 if 1.5 < t <= 2.0 else t, s, d) for t, s, d in playback]  # stall, then backlog

    _feed(assembler, mic, late)
    assembler.save(out)

    _assert_aligned(out, [1.0, 2.5])


# --- Microphone drops nobody reports (Windows/PyAudio, #21) ----------------

from voxscribe.track_assembler import DropDetector  # noqa: E402


def test_drop_detector_reports_samples_the_microphone_lost():
    detector = DropDetector(rate=16000)
    assert detector.missing(received=1600, now=10.1) == 0  # first read: 0.1 s of audio
    assert detector.missing(received=16000, now=11.0) == 0  # on time
    # 0.5 s later only 0.2 s more arrived: 0.3 s were lost.
    assert detector.missing(received=19200, now=11.5) == 4800


def test_drop_detector_ignores_jitter_and_slow_clock_drift():
    detector = DropDetector(rate=16000)
    detector.missing(received=160, now=0.01)
    assert detector.missing(received=160 + 15800, now=1.01) == 0     # 12 ms short: jitter
    # Device clock 50 ms per hour slower than the wall clock: far below the threshold.
    assert detector.missing(received=int(3600 * 16000 - 0.05 * 16000), now=3600.0) == 0


def test_drop_detector_counts_reported_padding_as_received():
    detector = DropDetector(rate=16000)
    detector.missing(received=1600, now=0.1)
    lost = detector.missing(received=1600 + 3200, now=0.6)  # 0.3 s lost
    assert lost == 4800
    assert detector.missing(received=1600 + 3200 + lost + 16000, now=1.6) == 0
