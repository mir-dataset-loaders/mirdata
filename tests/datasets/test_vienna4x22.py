"""Tests for the Vienna 4x22 loader."""

import logging
import os

try:
    import partitura
except ImportError:
    logging.error(
        "In order to test vienna4x22 you must have partitura installed. "
        "Please reinstall mirdata using `pip install 'mirdata[vienna4x22]'` "
        "and re-run the tests."
    )
    raise ImportError

import numpy as np

from mirdata.datasets import vienna4x22
from tests.test_utils import run_track_tests

DATA_HOME = os.path.normpath("tests/resources/mir_datasets/vienna4x22")
TRACK_ID = "Chopin_op10_no3_p01"


def test_track():
    dataset = vienna4x22.Dataset(DATA_HOME, version="test")
    track = dataset.track(TRACK_ID)

    expected_attributes = {
        "track_id": TRACK_ID,
        "piece": "Chopin_op10_no3",
        "pianist_id": "01",
        "alignment_quality": "manual",
        "audio_path": os.path.join(
            DATA_HOME, "audio/Chopin_Etude/Chopin_op10_no3_p01.wav"
        ),
        "score_path": os.path.join(DATA_HOME, "musicxml/Chopin_op10_no3.musicxml"),
        "performance_path": os.path.join(DATA_HOME, "midi/Chopin_op10_no3_p01.mid"),
        "match_path": os.path.join(DATA_HOME, "match/Chopin_op10_no3_p01.match"),
    }

    expected_property_types = {
        "audio": tuple,
        "score": partitura.score.Score,
        "performance": partitura.performance.Performance,
        "match": tuple,
        "score_note_array": np.ndarray,
        "performance_note_array": np.ndarray,
    }

    run_track_tests(track, expected_attributes, expected_property_types)


def test_load_audio():
    path = os.path.join(DATA_HOME, "audio/Chopin_Etude/Chopin_op10_no3_p01.wav")
    audio, sr = vienna4x22.load_audio(path)
    assert audio.shape == (2, 88200)  # stereo, 2 sec
    assert audio.dtype == np.float32
    assert sr == 44100
    assert vienna4x22.load_audio(None) is None


def test_load_score():
    path = os.path.join(DATA_HOME, "musicxml/Chopin_op10_no3.musicxml")
    score = vienna4x22.load_score(path)
    assert isinstance(score, partitura.score.Score)
    assert len(score.parts) == 1

    part = score[0]
    assert [(ks.fifths, ks.mode) for ks in part.key_sigs] == [(4, "major")]
    assert [(ts.beats, ts.beat_type) for ts in part.time_sigs] == [(2, 4)]

    na = score.note_array()
    assert len(na) == 486
    for field in ["onset_beat", "duration_beat", "pitch", "voice", "id"]:
        assert field in na.dtype.names

    # pickup note
    assert na[0]["id"] == "n1"
    assert na[0]["pitch"] == 59
    assert na[0]["onset_beat"] == -0.5
    assert na[0]["duration_beat"] == 0.5

    # grace note at the end
    assert na[-1]["id"] == "n450"
    assert na[-1]["onset_beat"] == 40.0
    assert na[-1]["duration_beat"] == 0.0

    assert vienna4x22.load_score(None) is None


def test_load_performance():
    path = os.path.join(DATA_HOME, "midi/Chopin_op10_no3_p01.mid")
    perf = vienna4x22.load_performance(path)
    assert isinstance(perf, partitura.performance.Performance)

    na = perf.note_array()
    assert len(na) == 451

    assert na[0]["onset_sec"] == 0.0
    assert na[0]["onset_tick"] == 0
    assert na[0]["duration_tick"] == 261
    assert na[0]["pitch"] == 59
    assert na[0]["velocity"] == 44

    assert na[1]["onset_tick"] == 678
    assert na[1]["pitch"] == 40
    assert na[1]["velocity"] == 22

    assert na[-1]["onset_tick"] == 78610
    assert na[-1]["pitch"] == 64

    # pedal events: 64 = sustain, 67 = soft
    controls = perf.performedparts[0].controls
    assert len([c for c in controls if c["number"] == 64]) == 3385
    assert len([c for c in controls if c["number"] == 67]) == 37

    assert vienna4x22.load_performance(None) is None


def test_load_match():
    path = os.path.join(DATA_HOME, "match/Chopin_op10_no3_p01.match")
    result = vienna4x22.load_match(path)
    assert isinstance(result, tuple)
    assert len(result) == 3
    performance, alignment, score = result
    assert isinstance(performance, partitura.performance.Performance)
    assert isinstance(alignment, list)
    assert isinstance(score, partitura.score.Score)

    labels = [a["label"] for a in alignment]
    assert labels.count("match") == 451
    assert labels.count("deletion") == 3
    assert labels.count("insertion") == 0

    deleted = [a["score_id"] for a in alignment if a["label"] == "deletion"]
    assert deleted == ["n356", "n359", "n454"]

    assert alignment[0] == {"label": "match", "score_id": "n1", "performance_id": "n0"}
    assert alignment[1] == {"label": "match", "score_id": "n2", "performance_id": "n2"}

    assert vienna4x22.load_match(None) is None


def test_score_performance_alignment():
    dataset = vienna4x22.Dataset(DATA_HOME, version="test")
    track = dataset.track(TRACK_ID)
    _, alignment, _ = track.match

    score_notes = {note["id"]: note for note in track.score_note_array}
    performance_notes = {note["id"]: note for note in track.performance_note_array}

    matches = [a for a in alignment if a["label"] == "match"]
    assert len(matches) == 451

    for a in matches:
        # every id in the alignment exists in the note arrays
        assert a["score_id"] in score_notes
        assert a["performance_id"] in performance_notes
        # a matched score note and performance note are the same key
        assert (
            score_notes[a["score_id"]]["pitch"]
            == performance_notes[a["performance_id"]]["pitch"]
        )
