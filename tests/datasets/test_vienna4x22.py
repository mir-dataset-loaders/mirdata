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
import pytest

from mirdata import annotations
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
        "matched_note_array": np.ndarray,
        "performance_notes": annotations.NoteData,
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


def test_performance_ids_follow_match():
    dataset = vienna4x22.Dataset(DATA_HOME, version="test")
    track = dataset.track(TRACK_ID)
    match_performance, _, _ = track.match

    def ids_by_key(performance):
        return {
            (note["note_on_tick"], note["midi_pitch"]): note["id"]
            for note in performance.performedparts[0].notes
        }

    # partitura numbers MIDI notes n0..n450, the .match file skips n448
    assert ids_by_key(track.performance) == ids_by_key(match_performance)
    assert "n448" not in track.performance_note_array["id"]
    assert "n451" in track.performance_note_array["id"]

    # pedal events still come from the MIDI file
    controls = track.performance.performedparts[0].controls
    assert len([c for c in controls if c["number"] == 64]) == 3385


def test_align_performance_ids():
    path = os.path.join(DATA_HOME, "midi/Chopin_op10_no3_p01.mid")
    match_path = os.path.join(DATA_HOME, "match/Chopin_op10_no3_p01.match")
    match_performance, _, _ = vienna4x22.load_match(match_path)

    assert vienna4x22.align_performance_ids(None, match_performance) is None

    performance = vienna4x22.load_performance(path)
    performance.performedparts[0].notes.pop()
    with pytest.raises(ValueError):
        vienna4x22.align_performance_ids(performance, match_performance)


def test_matched_note_array():
    dataset = vienna4x22.Dataset(DATA_HOME, version="test")
    track = dataset.track(TRACK_ID)
    matched = track.matched_note_array

    assert matched.dtype.names == (
        "score_id",
        "performance_id",
        "score_pitch",
        "performance_pitch",
        "onset_beat",
        "duration_beat",
        "onset_sec",
        "duration_sec",
        "velocity",
    )
    # 451 matches; the 3 deleted score notes are left out
    assert len(matched) == 451
    assert not {"n356", "n359", "n454"} & set(matched["score_id"])
    assert np.all(np.diff(matched["onset_sec"]) >= 0)

    first = matched[0]
    assert (first["score_id"], first["performance_id"]) == ("n1", "n0")
    assert first["score_pitch"] == first["performance_pitch"] == 59
    assert first["onset_beat"] == -0.5
    assert first["onset_sec"] == 0.0
    assert first["velocity"] == 44

    score_pitch = {n["id"]: n["pitch"] for n in track.score_note_array}
    assert all(score_pitch[m["score_id"]] == m["score_pitch"] for m in matched)
    assert np.array_equal(matched["score_pitch"], matched["performance_pitch"])


def test_performance_notes():
    dataset = vienna4x22.Dataset(DATA_HOME, version="test")
    track = dataset.track(TRACK_ID)
    notes = track.performance_notes

    assert notes.intervals.shape == (451, 2)
    assert notes.pitch_unit == "midi"
    assert notes.confidence_unit == "velocity"
    assert notes.pitches[0] == 59
    assert notes.confidence[0] == 44
    # offsets include the sustain pedal (partitura sound_off), key release is 0.27 s
    assert np.isclose(notes.intervals[0, 1], 0.873958)

    # row i of performance_notes is row i of performance_note_array, so the
    # note IDs can be recovered by position
    na = track.performance_note_array
    assert np.array_equal(notes.pitches, na["pitch"])
    assert np.array_equal(notes.confidence, na["velocity"])
    assert np.allclose(notes.intervals[:, 0], na["onset_sec"])


def test_performance_notes_recover_ids():
    dataset = vienna4x22.Dataset(DATA_HOME, version="test")
    track = dataset.track(TRACK_ID)
    notes = track.performance_notes
    ids = track.performance_note_array["id"]
    matched = {m["performance_id"]: m for m in track.matched_note_array}

    # every NoteData row recovers an ID that resolves to the same note in the
    # alignment
    for i in range(len(notes.pitches)):
        m = matched[ids[i]]
        assert notes.pitches[i] == m["performance_pitch"]
        assert notes.confidence[i] == m["velocity"]
        assert np.isclose(notes.intervals[i, 0], m["onset_sec"])

    # the note partitura numbers n448 from the MIDI file is n449 in the .match
    # file, aligned to score note n453 (G#3)
    i = int(np.flatnonzero(ids == "n449")[0])
    assert notes.pitches[i] == 56
    assert matched["n449"]["score_id"] == "n453"
    assert matched["n449"]["score_pitch"] == 56


def test_performance_notes_rows_must_line_up():
    dataset = vienna4x22.Dataset(DATA_HOME, version="test")
    track = dataset.track(TRACK_ID)

    # NoteData drops duplicate notes, which would shift every later row
    na = track.performance_note_array
    track.__dict__["performance_note_array"] = np.insert(na, 1, na[0])
    with pytest.raises(ValueError):
        track.performance_notes
