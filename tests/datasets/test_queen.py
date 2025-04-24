import os
import numpy as np

from mirdata import annotations
from mirdata.datasets import queen
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "0"
    data_home = os.path.normpath("tests/resources/mir_datasets/queen")
    dataset = queen.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/queen/"),
            "audio/Greatest Hits I/01 Bohemian Rhapsody.flac",
        ),
        "chords_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/queen/"),
            "annotations/chordlab/Queen/Greatest Hits I/01 Bohemian Rhapsody.lab",
        ),
        "keys_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/queen/"),
            "annotations/keylab/Queen/Greatest Hits I/01 Bohemian Rhapsody.lab",
        ),
        "sections_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/queen/"),
            "annotations/seglab/Queen/Greatest Hits I/01 Bohemian Rhapsody.lab",
        ),
        "title": "01 Bohemian Rhapsody",
        "track_id": "0",
    }

    expected_property_types = {
        "chords": annotations.ChordData,
        "key": annotations.KeyData,
        "sections": annotations.SectionData,
        "audio": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 44100, "sample rate {} is not 44100".format(sr)
    assert audio.shape == (144384,), "audio shape {} was not (144384,)".format(
        audio.shape
    )


def test_load_chords():
    chords_path = "tests/resources/mir_datasets/queen/annotations/chordlab/Queen/Greatest Hits I/01 Bohemian Rhapsody.lab"
    chord_data = queen.load_chords(chords_path)

    assert type(chord_data) == annotations.ChordData
    assert type(chord_data.intervals) == np.ndarray
    assert type(chord_data.labels) == list

    assert np.array_equal(chord_data.intervals[:, 0], np.array([0.0, 0.459, 4.122]))
    assert np.array_equal(chord_data.intervals[:, 1], np.array([0.459, 4.122, 4.911]))
    assert np.array_equal(chord_data.labels, ["N", "Bb:maj6", "C:7"])

    assert queen.load_chords(None) is None


def test_load_key():
    key_path = "tests/resources/mir_datasets/queen/annotations/keylab/Queen/Greatest Hits I/01 Bohemian Rhapsody.lab"
    key_data = queen.load_key(key_path)

    assert type(key_data) == annotations.KeyData
    assert type(key_data.intervals) == np.ndarray

    assert np.array_equal(key_data.intervals[:, 0], np.array([0.456, 83.139]))
    assert np.array_equal(key_data.intervals[:, 1], np.array([83.139, 108.519]))
    assert np.array_equal(key_data.keys, ["Bb", "Eb"])

    assert queen.load_key(None) is None


def test_load_sections():
    sections_path = "tests/resources/mir_datasets/queen/annotations/seglab/Queen/Greatest Hits I/01 Bohemian Rhapsody.lab"
    section_data = queen.load_sections(sections_path)

    assert type(section_data) == annotations.SectionData
    assert type(section_data.intervals) == np.ndarray
    assert type(section_data.labels) == list

    assert np.array_equal(
        section_data.intervals[:, 0], np.array([0.0, 0.4, 49.072, 108.392])
    )
    assert np.array_equal(
        section_data.intervals[:, 1], np.array([0.4, 49.072, 108.392, 156.32])
    )
    assert np.array_equal(
        section_data.labels, np.array(["silence", "intro", "verse", "verse"])
    )

    assert queen.load_sections(None) is None
