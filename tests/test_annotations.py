import sys
import pytest
import numpy as np

import mirdata
from mirdata import annotations


def test_repr():
    class TestAnnotation(annotations.Annotation):
        def __init__(self):
            self.a = ["a", "b", "c"]
            self.b = np.array([[1, 2], [1, 4]])
            self._c = "hidden"

    test_track = TestAnnotation()
    assert test_track.__repr__() == """TestAnnotation(a, b)"""

    beat_data = annotations.BeatData(
        np.array([1.0, 2.0]), "s", np.array([1, 2]), "bar_index"
    )
    assert (
        beat_data.__repr__() == "BeatData(position_unit, positions, time_unit, times)"
    )


def test_beat_data():
    times = np.array([1.0, 2.0])
    positions = np.array([3, 4])
    beat_data = annotations.BeatData(times, "s", positions, "bar_index")
    assert np.allclose(beat_data.times, times)
    assert np.allclose(beat_data.positions, positions)

    with pytest.raises(ValueError):
        annotations.BeatData(None, None, None, None)

    with pytest.raises(TypeError):
        annotations.BeatData([1.0, 2.0], "s", [1, 2], "bar_index")

    with pytest.raises(TypeError):
        annotations.BeatData(times, "s", [3, 4], "bar_index")

    with pytest.raises(TypeError):
        annotations.BeatData(times.astype(int), "s", [3, 4], "bar_index")

    with pytest.raises(ValueError):
        annotations.BeatData(times, "s", np.array([1]), "bar_index")


def test_section_data():
    intervals = np.array([[1.0, 2.0], [1.5, 3.0], [2.0, 3.0]])
    labels = ["a", "b", "c"]
    section_data = annotations.SectionData(intervals, "s")
    assert np.allclose(section_data.intervals, intervals)
    assert section_data.labels is None
    section_data2 = annotations.SectionData(intervals, "s", labels, "open")
    assert np.allclose(section_data2.intervals, intervals)
    assert section_data2.labels == labels

    with pytest.raises(ValueError):
        annotations.SectionData(None, None)

    with pytest.raises(TypeError):
        annotations.SectionData([1.0, 2.0], "s")

    with pytest.raises(TypeError):
        annotations.SectionData(intervals, "s", np.array(labels), "open")

    with pytest.raises(TypeError):
        annotations.SectionData(intervals.astype(int), "s")

    with pytest.raises(ValueError):
        annotations.SectionData(intervals, "s", ["a"], "open")

    with pytest.raises(ValueError):
        annotations.SectionData(np.array([1.0, 2.0, 3.0]), "s")

    with pytest.raises(ValueError):
        annotations.SectionData(np.array([[1.0, 2.0], [2.0, 1.0]]), "s")


def test_note_data():
    intervals = np.array([[1.0, 2.0], [1.5, 3.0], [2.0, 3.0]])
    notes = np.array([100.0, 150.0, 120.0])
    confidence = np.array([0.1, 0.4, 0.2])
    note_data = annotations.NoteData(intervals, "s", notes, "hz")
    assert np.allclose(note_data.intervals, intervals)
    assert np.allclose(note_data.notes, notes)
    assert note_data.confidence is None
    note_data2 = annotations.NoteData(
        intervals, "s", notes, "hz", confidence, "likelihood"
    )
    assert np.allclose(note_data2.intervals, intervals)
    assert np.allclose(note_data2.notes, notes)
    assert np.allclose(note_data2.confidence, confidence)

    with pytest.raises(ValueError):
        annotations.NoteData(None, "s", notes, "hz")

    with pytest.raises(ValueError):
        annotations.NoteData(intervals, "s", None, "hz")

    with pytest.raises(TypeError):
        annotations.NoteData([1.0, 2.0], "s", notes, "hz")

    with pytest.raises(TypeError):
        annotations.NoteData(intervals, "s", [3.0, 4.0], "hz")

    with pytest.raises(TypeError):
        annotations.NoteData(intervals.astype(int), "s", notes, "hz")

    with pytest.raises(ValueError):
        annotations.NoteData(intervals, "s", np.array([1.0]), "hz")


def test_chord_data():
    intervals = np.array([[1.0, 2.0], [1.5, 3.0], [2.0, 3.0]])
    labels = ["E:min", "A", "G:7"]
    confidence = np.array([0.1, 0.4, 0.2])
    chord_data = annotations.ChordData(
        intervals, "s", labels, "harte", confidence, "likelihood"
    )
    assert np.allclose(chord_data.intervals, intervals)
    assert chord_data.labels == labels
    assert np.allclose(chord_data.confidence, confidence)


def test_f0_data():
    times = np.array([1.0, 2.0, 3.0])
    frequencies = np.array([100.0, 150.0, 120.0])
    confidence = np.array([0.1, 0.4, 0.2])
    f0_data = annotations.F0Data(
        times, "s", frequencies, "hz", confidence, "likelihood"
    )
    assert np.allclose(f0_data.times, times)
    assert np.allclose(f0_data.frequencies, frequencies)
    assert np.allclose(f0_data.confidence, confidence)


def test_multif0_data():
    times = np.array([1.0, 2.0])
    frequencies = [[100.0], [150.0, 120.0]]
    confidence = [[0.1], [0.4, 0.2]]
    f0_data = annotations.MultiF0Data(
        times, "s", frequencies, "hz", confidence, "likelihood"
    )
    assert np.allclose(f0_data.times, times)
    assert f0_data.frequency_list is not None
    assert f0_data.confidence_list is not None


def test_key_data():
    intervals = np.array([[1.0, 2.0], [1.5, 3.0], [2.0, 3.0]])
    keys = ["E:minor", "A", "G"]
    key_data = annotations.KeyData(intervals, "s", keys, "key_mode")
    assert np.allclose(key_data.intervals, intervals)
    assert key_data.keys == keys


def test_lyric_data():
    intervals = np.array([[1.0, 2.0], [1.5, 3.0], [2.0, 3.0]])
    lyrics = ["E:m", "A", "G:7"]
    lyric_data = annotations.LyricData(intervals, "s", lyrics, "words")
    assert np.allclose(lyric_data.intervals, intervals)
    assert lyric_data.lyrics == lyrics


def test_tempo_data():
    intervals = np.array([[1.0, 2.0], [1.5, 3.0], [2.0, 3.0]])
    value = np.array([140.0, 110.0, 111.0])
    confidence = np.array([0.1, 0.4, 0.2])
    tempo_data = annotations.TempoData(
        intervals, "s", value, "bpm", confidence, "likelihood"
    )
    assert np.allclose(tempo_data.intervals, intervals)
    assert np.allclose(tempo_data.value, value)
    assert np.allclose(tempo_data.confidence, confidence)


def test_event_data():
    intervals = np.array([[1.0, 2.0], [1.5, 3.0], [2.0, 3.0]])
    events = ["E:m", "A", "G:7"]
    event_data = annotations.EventData(intervals, "s", events, "open")
    assert np.allclose(event_data.intervals, intervals)
    assert event_data.events == events


def test_validate_array_like():
    with pytest.raises(ValueError):
        annotations.validate_array_like(None, list, str)

    annotations.validate_array_like(None, list, str, none_allowed=True)

    with pytest.raises(TypeError):
        annotations.validate_array_like([1, 2], np.ndarray, str)

    with pytest.raises(TypeError):
        annotations.validate_array_like([1, 2], list, str)

    with pytest.raises(TypeError):
        annotations.validate_array_like(np.array([1, 2]), np.ndarray, str)

    with pytest.raises(ValueError):
        annotations.validate_array_like([], list, int)


def test_validate_lengths_equal():
    annotations.validate_lengths_equal([np.array([0, 1])])
    annotations.validate_lengths_equal([np.array([]), None])

    with pytest.raises(ValueError):
        annotations.validate_lengths_equal([np.array([0, 1]), np.array([0])])


def test_validate_confidence():
    annotations.validate_confidence(None, None)

    with pytest.raises(ValueError):
        annotations.validate_confidence(np.array([[0, 1], [0, 2]]), "likelihood")
    with pytest.raises(ValueError):
        annotations.validate_confidence(np.array([0, 2]), "binary")


def test_validate_times():
    annotations.validate_times(None, None)

    with pytest.raises(ValueError):
        annotations.validate_times(np.array([[0, 1], [0, 2]]), "s")

    with pytest.raises(ValueError):
        annotations.validate_times(np.array([2, 0]), "s")


def test_validate_intervals():
    annotations.validate_intervals(None, None)

    with pytest.raises(ValueError):
        annotations.validate_intervals(np.array([0, 2]), "s")

    with pytest.raises(ValueError):
        annotations.validate_intervals(np.array([0, -2]), "s")

    with pytest.raises(ValueError):
        annotations.validate_intervals(np.array([[0, 1], [1, 0.5]]), "s")
