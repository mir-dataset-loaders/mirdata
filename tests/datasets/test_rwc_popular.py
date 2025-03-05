import os
import numpy as np

from mirdata.datasets import rwc_popular
from mirdata import annotations
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "RM-P001"
    data_home = os.path.normpath("tests/resources/mir_datasets/rwc_popular")
    dataset = rwc_popular.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "RM-P001",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_popular/"),
            "audio/rwc-p-m01/1.wav",
        ),
        "sections_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_popular/"),
            "annotations/AIST.RWC-MDB-P-2001.CHORUS/RM-P001.CHORUS.TXT",
        ),
        "beats_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_popular/"),
            "annotations/AIST.RWC-MDB-P-2001.BEAT/RM-P001.BEAT.TXT",
        ),
        "chords_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_popular/"),
            "annotations/AIST.RWC-MDB-P-2001.CHORD/RWC_Pop_Chords/N001-M01-T01.lab",
        ),
        "voca_inst_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_popular/"),
            "annotations/AIST.RWC-MDB-P-2001.VOCA_INST/RM-P001.VOCA_INST.TXT",
        ),
        "piece_number": "No. 1",
        "suffix": "M01",
        "track_number": "Tr. 01",
        "title": "Eien no replica",
        "artist": "Kazuo Nishi",
        "singer_information": "Male",
        "duration": 209,
        "tempo": "135",
        "instruments": "Gt",
        "drum_information": "Drum sequences",
    }

    expected_property_types = {
        "beats": annotations.BeatData,
        "sections": annotations.SectionData,
        "chords": annotations.ChordData,
        "vocal_instrument_activity": annotations.EventData,
        "audio": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    y, sr = track.audio
    assert sr == 44100
    assert y.shape == (44100 * 2,)


def test_load_chords():
    chords_path = (
        "tests/resources/mir_datasets/rwc_popular/"
        + "annotations/AIST.RWC-MDB-P-2001.CHORD/RWC_Pop_Chords/N001-M01-T01.lab"
    )
    chord_data = rwc_popular.load_chords(chords_path)

    # check types
    assert type(chord_data) is annotations.ChordData
    assert type(chord_data.intervals) is np.ndarray
    assert type(chord_data.labels) is list

    # check values
    assert np.array_equal(
        chord_data.intervals[:, 0], np.array([0.000, 0.104, 3.646, 43.992, 44.494])
    )
    assert np.array_equal(
        chord_data.intervals[:, 1], np.array([0.104, 1.858, 5.387, 44.494, 47.636])
    )
    assert np.array_equal(
        chord_data.labels, ["N", "Ab:min", "E:maj", "Bb:maj(*3)", "C:min7"]
    )


def test_load_vocal_activity():
    vocinst_path = (
        "tests/resources/mir_datasets/rwc_popular/"
        + "annotations/AIST.RWC-MDB-P-2001.VOCA_INST/RM-P001.VOCA_INST.TXT"
    )
    vocinst_data = rwc_popular.load_vocal_activity(vocinst_path)

    # check types
    assert type(vocinst_data) is annotations.EventData
    assert type(vocinst_data.intervals) is np.ndarray
    assert type(vocinst_data.events) is list

    # check values
    assert np.array_equal(
        vocinst_data.intervals[:, 0],
        np.array(
            [
                0.000,
                10.293061224,
                11.883492063,
                12.087845804,
                13.587460317,
                13.819387755,
                20.668707482,
                20.832653061,
            ]
        ),
    )
    assert np.array_equal(
        vocinst_data.intervals[:, 1],
        np.array(
            [
                10.293061224,
                11.883492063,
                12.087845804,
                13.587460317,
                13.819387755,
                20.668707482,
                20.832653061,
                26.465306122,
            ]
        ),
    )
    assert np.array_equal(
        vocinst_data.events,
        np.array(
            ["b", "m:withm", "b", "m:withm", "b", "m:withm", "b", "s:electricguitar"]
        ),
    )


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/rwc_popular"
    dataset = rwc_popular.Dataset(data_home, version="test")
    metadata = dataset._metadata
    assert metadata["RM-P001"] == {
        "piece_number": "No. 1",
        "suffix": "M01",
        "track_number": "Tr. 01",
        "title": "Eien no replica",
        "artist": "Kazuo Nishi",
        "singer_information": "Male",
        "duration": 209,
        "tempo": "135",
        "instruments": "Gt",
        "drum_information": "Drum sequences",
    }
