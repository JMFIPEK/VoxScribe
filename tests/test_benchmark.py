"""Benchmark (#56): word error rate, word timings, and the runner with fake
Backends - no real model, nothing downloaded."""

import numpy as np
import pytest
import soundfile as sf

from voxscribe import benchmark as b
from voxscribe import tracks
from voxscribe.transcription.registry import BackendRegistry

SR = 16000


# --- Word error rate -----------------------------------------------------------

def test_identical_text_has_no_errors():
    e = b.word_errors("Hallo, hörst du mich gut?", "hallo hörst du mich gut")
    assert (e.rate, e.reference_words) == (0.0, 5)


def test_errors_are_split_into_substitutions_deletions_insertions():
    e = b.word_errors("das ist ein test", "das war ein kleiner")
    assert (e.substitutions, e.deletions, e.insertions) == (2, 0, 0)
    e = b.word_errors("das ist ein test", "das ist test")
    assert (e.substitutions, e.deletions, e.insertions) == (0, 1, 0)
    e = b.word_errors("das ist ein test", "das ist ein kurzer test")
    assert (e.substitutions, e.deletions, e.insertions) == (0, 0, 1)
    assert e.rate == pytest.approx(0.25)


def test_case_and_punctuation_dont_count_but_different_numbers_do():
    assert b.word_errors("It's 8 o'clock.", "it’s 8 O'CLOCK").rate == 0.0
    assert b.word_errors("seit acht Uhr", "seit 9 Uhr").substitutions == 1


def test_empty_hypothesis_deletes_every_word():
    e = b.word_errors("eins zwei drei", "")
    assert (e.deletions, e.rate) == (3, 1.0)


# --- Word timings and merging --------------------------------------------------

def test_timing_offset_is_the_median_over_matching_words():
    own = [{"word": "Hallo,", "start": 0.10}, {"word": "du", "start": 0.60}, {"word": "da", "start": 1.00}]
    aligned = [{"word": "hallo", "start": 0.00}, {"word": "du", "start": 0.50}, {"word": "da", "start": 0.70}]
    assert b.word_timing_offset(own, aligned) == pytest.approx(0.10)


def test_timing_offset_is_none_without_common_words():
    assert b.word_timing_offset([{"word": "a", "start": 0}], [{"word": "b", "start": 0}]) is None


def test_separate_tracks_merge_in_time_order():
    mic = [{"start": 0.0, "text": "Hallo"}, {"start": 4.0, "text": "Gut."}]
    playback = [{"start": 2.0, "text": "Hi."}]
    assert b.transcript_text(b.merge_by_time(mic, playback)) == "Hallo Hi. Gut."


# --- Runner ----------------------------------------------------------------------

class FakeBackend:
    """Transcribes a file as one segment per second of audio, words included
    if `words`; `fail` raises instead."""

    def __init__(self, name, text="eins zwei drei", words=False, skips_alignment=False, fail=None):
        self.name, self.text, self.words, self.fail = name, text, words, fail
        self.skips_alignment = skips_alignment
        self.calls = []

    def describe(self, model_id, device):
        return ""

    def transcribe(self, request):
        if self.fail:
            print("helper said: no permission")
            raise RuntimeError(self.fail)
        samples = request.audio.load()
        self.calls.append(len(samples) / SR)
        seg = {"start": 0.0, "end": len(samples) / SR, "text": self.text}
        if self.words:
            seg["words"] = [{"word": w, "start": 0.1 * i + 0.05, "end": 0.1 * i + 0.1}
                            for i, w in enumerate(self.text.split())]
        return {"segments": [seg], "language": "de"}


def fake_align(result, path, language, device):
    segs = []
    for s in result["segments"]:
        words = [{"word": w, "start": 0.1 * i, "end": 0.1 * i + 0.05} for i, w in enumerate(s["text"].split())]
        segs.append(dict(s, words=words))
    return {"segments": segs, "language": language}


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "rec.wav"
    sf.write(path, (0.1 * np.sin(np.arange(5 * SR) / 10)).astype(np.float32), SR)
    return str(path)


def make_bench(backends, presence=lambda m: None, align_presence=lambda lang: None, logs=None):
    registry = BackendRegistry()
    for prefix, backend in backends.items():
        registry.register(prefix, lambda backend=backend: backend)
    return b.Benchmark(registry=registry, presence=presence, align=fake_align,
                       align_presence=align_presence, device="cpu",
                       log=(logs.append if logs is not None else lambda line: None))


def test_each_model_gets_a_row_with_wer_and_times(wav):
    good = FakeBackend("good", "eins zwei drei")
    bad = FakeBackend("bad", "eins drei")
    bench = make_bench({"good:": good, "bad:": bad})
    recs, skipped = bench.run([wav], ["eins zwei drei"], ["good:x", "bad:x"], seconds=0)

    runs = {r.model_id: r for r in recs[0].runs}
    assert runs["good:x"].errors.rate == 0.0
    assert runs["bad:x"].errors.deletions == 1
    assert runs["good:x"].audio_s == pytest.approx(5.0)
    assert runs["good:x"].load_s is not None and runs["good:x"].align_s is not None
    assert runs["good:x"].realtime > 0
    assert skipped == []


def test_the_warm_up_run_is_one_second_and_seconds_cut_the_audio(wav):
    backend = FakeBackend("x")
    make_bench({"x:": backend}).run([wav], [None], ["x:1"], seconds=2)
    assert backend.calls == [pytest.approx(1.0), pytest.approx(2.0)]


def test_a_model_not_on_the_machine_is_skipped_and_never_run(wav):
    backend = FakeBackend("x")
    bench = make_bench({"x:": backend}, presence=lambda m: "not downloaded")
    recs, skipped = bench.run([wav], [None], ["x:big"])
    assert skipped == [("x:big", "not downloaded")]
    assert recs[0].runs == [] and backend.calls == []


def test_download_runs_it_anyway(wav):
    backend = FakeBackend("x")
    bench = make_bench({"x:": backend}, presence=lambda m: "not downloaded")
    recs, skipped = bench.run([wav], [None], ["x:big"], download=True)
    assert skipped == [] and len(recs[0].runs) == 1


def test_a_failing_backend_only_fails_its_own_row(wav):
    logs = []
    bench = make_bench({"ok:": FakeBackend("ok"), "broken:": FakeBackend("broken", fail="exit 1")}, logs=logs)
    recs, _ = bench.run([wav], ["eins zwei drei"], ["broken:x", "ok:x"])
    runs = {r.model_id: r for r in recs[0].runs}
    assert runs["broken:x"].status == "failed: exit 1" and runs["broken:x"].errors is None
    assert runs["ok:x"].status == "ok" and runs["ok:x"].errors.rate == 0.0
    assert any("no permission" in line for line in logs)  # its output, shown on failure


def test_a_backend_with_own_word_timings_reports_its_offset_and_isnt_aligned(wav):
    backend = FakeBackend("apple", words=True, skips_alignment=True)
    recs, _ = make_bench({"apple:": backend}).run([wav], [None], ["apple:x"])
    run = recs[0].runs[0]
    assert run.align_s is None
    assert run.timing_offset_s == pytest.approx(0.05)
    assert "words" in run.returns


def test_missing_alignment_model_keeps_the_text_and_says_so(wav):
    bench = make_bench({"x:": FakeBackend("x")}, align_presence=lambda lang: "not downloaded")
    recs, _ = bench.run([wav], ["eins zwei drei"], ["x:1"])
    run = recs[0].runs[0]
    assert run.status.startswith("ok, not aligned") and run.errors.rate == 0.0


def test_a_call_is_compared_as_mix_and_as_separate_tracks(tmp_path):
    path = str(tmp_path / "call.wav")
    mic = (0.2 * np.sin(np.arange(3 * SR) / 7)).astype(np.float32)
    system = (0.2 * np.sin(np.arange(3 * SR) / 5)).astype(np.float32)
    sf.write(path, tracks.mix(mic, system, full_scale=1.0), SR)
    tracks.save_tracks(path, mic * 32767, system * 32767, setup="headset")

    backend = FakeBackend("x", text="hallo")
    recs, _ = make_bench({"x:": backend}).run([path], [None], ["x:1"], tracks="both", seconds=0)
    modes = [r.mode for r in recs[0].runs]
    assert modes == ["mix", "separate"]
    separate = recs[0].runs[1]
    assert separate.text == "hallo hallo"  # one segment per track, merged
    assert separate.returns == "1+1 segments"
    # warm-up + mix, then warm-up + both tracks
    assert backend.calls == [pytest.approx(1.0), pytest.approx(3.0),
                             pytest.approx(1.0), pytest.approx(3.0), pytest.approx(3.0)]


def test_separate_on_a_plain_recording_falls_back_to_the_mix(wav):
    logs = []
    recs, _ = make_bench({"x:": FakeBackend("x")}, logs=logs).run([wav], [None], ["x:1"], tracks="separate")
    assert [r.mode for r in recs[0].runs] == ["mix"]
    assert any("not a Call" in line for line in logs)


def test_report_has_a_table_the_texts_and_the_skipped_models(wav):
    recs, skipped = make_bench({"x:": FakeBackend("x")}).run([wav], ["eins zwei drei"], ["x:1"])
    report = b.format_report(recs, skipped + [("big", "not downloaded")], machine="Machine: test")
    assert report.startswith("Machine: test")
    assert "| x:1 " in report and "0.0%" in report
    assert "### x:1 (mix)\neins zwei drei" in report
    assert "big (not downloaded)" in report


# --- Numbers: written as words or digits count the same (#58) -----------------

@pytest.mark.parametrize("words, digits", [
    ("seit acht Uhr", "seit 8 Uhr"),
    ("insgesamt siebenundzwanzig Aufgaben", "insgesamt 27 Aufgaben"),
    ("zwölf davon", "12 davon"),
    ("about eighty percent", "about 80%"),
    ("about eighty percent", "about 80 %"),
    ("twenty-seven tasks", "27 tasks"),
    ("ten minutes", "10 minutes"),
    ("etwa zwanzig Prozent", "etwa 20 %"),
])
def test_numbers_as_words_or_digits_are_the_same_word(words, digits):
    assert b.word_errors(words, digits).rate == 0.0


def test_articles_are_not_numbers():
    # "ein"/"one" stay words: a model writing "1" for "one option" is wrong.
    assert b.word_errors("one option", "1 option").substitutions == 1
    assert b.word_errors("ein neues Werkzeug", "1 neues Werkzeug").substitutions == 1
