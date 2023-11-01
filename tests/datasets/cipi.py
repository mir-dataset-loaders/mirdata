import logging
import os

try:
    import music21
except ImportError:
    logging.error(
        "In order to test haydn_op20 you must have music21 installed. "
        "Please reinstall mirdata using `pip install 'mirdata[haydn_op20] and re-run the tests."
    )
    raise ImportError

from mirdata.annotations import KeyData, ChordData
from mirdata.datasets import cipi
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "cipi_c-181"
    data_home = os.path.normpath("tests/resources/mir_datasets/cipi")
    dataset = cipi.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        {
            "annotations": {
                "lh_fingering": [
                    "ArGNNThumb-s/lh/c-181.pt",
                    "90fe1f62d8f3dc191569336e7e4faefd",
                ],
                "rh_fingering": [
                    "ArGNNThumb-s/rh/c-181.pt",
                    "72a0c575c2883826a2dbfa7d609071e3",
                ],
                "expressiviness": [
                    "virtuoso/c-181.pt",
                    "6ab15c794356bc3bd58c1fb089455f03",
                ],
                "notes": ["k/c-181.pt", "b34227117c32a4b78a2255fdd9d5fa9f"],
            }
        }
    }

    expected_property_types = {
        "title": str,
        "book": str,
        "URI": str,
        "composer": str,
        "track_id": str,
        "musicxml_paths": list,
        "difficulty_annotation": str,
        "scores": list,
        "fingering": tuple,
        "expressiviness": list,
        "notes": list,
    }
    run_track_tests(track, expected_attributes, expected_property_types)


def test_to_jam():
    data_home = os.path.normpath("tests/resources/mir_datasets/cipi")
    dataset = cipi.Dataset(data_home)
    track = dataset.track("c_181")
    jam = track.to_jams()
    assert (
        jam["file_metadata"]["title"] == "12 Piano Variations on a Minuet KV 179 (189a)"
    ), "title does not match expected"
    assert (
        jam["file_metadata"]["artist"] == "WOLFGANG AMADEUS MOZART"
    ), "artist does not match expected"
    assert (
        jam["file_metadata"]["composer"] == "WOLFGANG AMADEUS MOZART"
    ), "composer does not match expected"
    assert jam["sandbox"]["book"] == "Piano Variations", "book does not match expected"
    assert (
        jam["sandbox"]["URI"]
        == "https://www.henle.de/en/detail/?Title=Piano+Variations_116"
    ), "book does not match expected"
    assert (
        jam["sandbox"]["difficulty_annotation"] == 5
    ), "difficulty_annotation does not match expected"
    assert jam["sandbox"]["duration"] == 0, "duration does not match expected"
    assert jam["sandbox"]["musicxml_paths"] == [
        "craig_files/TAVERN-master/Mozart/K179/Krn/K179.musicxml"
    ], "musicxml_paths does not match expected"


def test_load_score():
    path = os.path.normpath("craig_files/scarlatti-keyboard-sonatas-master/kern/L334K122.musicxml")
    data_home = os.path.normpath("tests/resources/mir_datasets/cipi")
    score = cipi.load_score(path, data_home)
    assert isinstance(score, music21.stream.Score)
    assert len(score.parts) == 2


def test_load_midi_path():
    path = os.path.normpath("craig_files/scarlatti-keyboard-sonatas-master/kern/L334K122.musicxml")
    data_home = os.path.normpath("tests/resources/mir_datasets/cipi")
    midi_path = cipi.convert_and_save_to_midi(path, data_home)
    assert isinstance(midi_path, str)
    assert (
        midi_path
        == "tests/resources/mir_datasets/cipi/craig_files/scarlatti-keyboard-sonatas-master/kern/L334K122.mid"
    )
