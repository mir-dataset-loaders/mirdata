import music21

from mirdata.datasets import haydn_op20
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "0"
    data_home = "tests/resources/mir_datasets/haydn_op20"
    dataset = haydn_op20.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "humdrum_annotated_path": "tests/resources/mir_datasets/haydn_op20/op20n1-01.hrm",
        "title": "op20n1-01",
        "track_id": "0",
    }

    expected_property_types = {
        "duration": int,
        "chords": list,
        "roman_numerals": list,
        "keys": list,
        "score": music21.stream.Score,
        "midi_path": str,
    }
    run_track_tests(track, expected_attributes, expected_property_types)


def test_to_jam():
    data_home = "tests/resources/mir_datasets/haydn_op20"
    dataset = haydn_op20.Dataset(data_home)
    track = dataset.track("0")
    jam = track.to_jams()
    assert jam["file_metadata"]["title"] == "op20n1-01", "title does not match expected"
    assert jam["file_metadata"]["duration"] == 12152, "duration does not match expected"
    assert (
        jam["sandbox"]["humdrum_annotated_path"]
        == "tests/resources/mir_datasets/haydn_op20/op20n1-01.hrm"
    ), "duration does not match expected"
    assert (
        jam["sandbox"]["midi_path"]
        == "tests/resources/mir_datasets/haydn_op20/op20n1-01.midi"
    ), "duration does not match expected"
    assert type(jam["sandbox"]["chords"]) == list
    assert type(jam["sandbox"]["key"]) == list
    assert type(jam["sandbox"]["roman_numerals"]) == list


def test_load_score():
    path = "tests/resources/mir_datasets/haydn_op20/op20n1-01.hrm"
    score = haydn_op20.load_score(path)
    assert type(score) == music21.stream.Score


def test_load_key():
    path = "tests/resources/mir_datasets/haydn_op20/op20n1-01.hrm"
    key = haydn_op20.load_key(path)
    assert type(key) == list
    assert len(key) == 156
    assert key[0]["time"] == 0
    assert key[-1]["time"] == 12152
    assert type(key[0]["key"]) == music21.key.Key


def test_load_chords():
    path = "tests/resources/mir_datasets/haydn_op20/op20n1-01.hrm"
    chords = haydn_op20.load_chords(path)
    assert type(chords) == list
    assert len(chords) == 156
    assert chords[0]["time"] == 0
    assert chords[-1]["time"] == 12152
    assert chords[0]["chord"] == "Eb-major triad"
    assert chords[-1]["chord"] == "Eb-major triad"


def test_load_roman_numerals():
    path = "tests/resources/mir_datasets/haydn_op20/op20n1-01.hrm"
    roman_numerals = haydn_op20.load_roman_numerals(path)
    assert type(roman_numerals) == list
    assert len(roman_numerals) == 156
    assert roman_numerals[0]["time"] == 0
    assert roman_numerals[-1]["time"] == 12152
    assert roman_numerals[0]["roman_numeral"] == "I"
    assert roman_numerals[-1]["roman_numeral"] == "I"


def test_load_midi_path():
    path = "tests/resources/mir_datasets/haydn_op20/op20n1-01.hrm"
    midi_path = haydn_op20.load_midi_path(path)
    assert type(midi_path) == str
    assert midi_path == "tests/resources/mir_datasets/haydn_op20/op20n1-01.midi"
