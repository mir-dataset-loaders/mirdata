# -*- coding: utf-8 -*-

import numpy as np
from mirdata import annotations
from mirdata.datasets import compmusic_carnatic_varnam
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "dharini_abhogi"
    data_home = "tests/resources/mir_datasets/compmusic_carnatic_varnam"
    track = compmusic_carnatic_varnam.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        "track_id": "dharini_abhogi",
        "audio_path": "tests/resources/mir_datasets/compmusic_carnatic_varnam/carnatic_varnam_1.0/"
        + "Audio/223578__gopalkoduri__carnatic-varnam-by-dharini-in-abhogi-raaga.mp3",
        "taala_path": "tests/resources/mir_datasets/compmusic_carnatic_varnam/carnatic_varnam_1.0/"
        + "Notations_Annotations/annotations/taalas/abhogi/dharini.svl",
        "notation_path": "tests/resources/mir_datasets/compmusic_carnatic_varnam/carnatic_varnam_1.0/"
        + "Notations_Annotations/notations/abhogi.yaml",
        "metadata_path": "tests/resources/mir_datasets/compmusic_carnatic_varnam/carnatic_varnam_1.0/"
        + "Notations_Annotations/annotations/tonics.yaml",
        "tonic": 200.58,
        "raaga": 'abhogi',
        "artist": 'dharini',
    }

    expected_property_types = {
        "audio": (np.ndarray, float),
        "taala": annotations.BeatData,
        "notation": annotations.EventData,
        "sections": annotations.SectionData,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 22100


def test_to_jams():
    default_trackid = "dharini_abhogi"
    data_home = "tests/resources/mir_datasets/compmusic_carnatic_varnam"
    track = compmusic_carnatic_varnam.Track(default_trackid, data_home=data_home)
    jam = track.to_jams()

    # Check metadata information
    assert jam["sandbox"].tonic == 200.58
    assert jam["sandbox"].performer == 'dharini'
    assert jam["sandbox"].raaga == 'abhogi'

    # Taala
    taala = jam.search(namespace="beat")[0]["data"]
    assert len(taala) == 384
    assert [i.time for i in taala[:4]] == [
        3.1299319727891155, 3.7754648526077097, 4.519818594104309, 5.188299319727891
    ]
    assert [i.duration for i in taala[:4]] == [0.0, 0.0, 0.0, 0.0]
    assert [i.value for i in taala[:4]] == [1, 1, 1, 1]
    assert [i.confidence for i in taala[:4]] == [None, None, None, None]

    # Sections
    sections = jam.search(namespace="segment_open")[0]["data"]
    print(sections)
    assert [section.time for section in sections] == [
        27.148526077097507,
        106.39496598639455,
        171.28802721088437,
        226.56222222222223,
        248.46607709750566,
    ]
    assert [section.duration for section in sections] == [
        78.60299319727892,
        64.20380952380954,
        54.63414965986394,
        21.16226757369614,
        92.58562358276646,
    ]
    assert [section.value for section in sections] == [
        'pallavi', 'anupallavi', 'muktayiswaram', 'charanam', 'chittiswaram'
    ]
    assert [section.confidence for section in sections] == [None, None, None, None, None]

    # Notation
    notation = jam.search(namespace="tag_open")[0]["data"]
    assert [note.time for note in notation[:4]] == [0.0, 3.1299319727891155, 3.7754648526077097, 4.519818594104309]
    assert [note.duration for note in notation[:4]] == [
        3.1299319727891155,
        0.6455328798185942,
        0.744353741496599,
        0.6684807256235823
    ]
    assert [note.value for note in notation[:4]] == ['R', ',', 'G', ',']
    assert [note.confidence for note in notation[:4]] == [None, None, None, None]


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/compmusic_carnatic_varnam"
    track = compmusic_carnatic_varnam.Track("dharini_abhogi", data_home=data_home)
    parsed_tonic = track.tonic
    assert parsed_tonic == 200.58


def test_load_taala():
    data_home = "tests/resources/mir_datasets/compmusic_carnatic_varnam"
    track = compmusic_carnatic_varnam.Track("dharini_abhogi", data_home=data_home)
    taala_path = track.taala_path
    parsed_taala = compmusic_carnatic_varnam.load_taala(taala_path)

    # Check types
    assert type(parsed_taala) == annotations.BeatData
    assert type(parsed_taala.times) is np.ndarray
    assert type(parsed_taala.positions) is np.ndarray

    # Check values
    assert np.array_equal(parsed_taala.times[:4], np.array([
        3.1299319727891155, 3.7754648526077097, 4.519818594104309, 5.188299319727891
    ]))
    assert np.array_equal(parsed_taala.positions[:4], np.array([1, 1, 1, 1]))
    assert compmusic_carnatic_varnam.load_taala(None) is None


def test_load_notation():
    data_home = "tests/resources/mir_datasets/compmusic_carnatic_varnam"
    track = compmusic_carnatic_varnam.Track("dharini_abhogi", data_home=data_home)
    notation_path = track.notation_path
    taala_path = track.taala_path
    parsed_notation = compmusic_carnatic_varnam.load_notation(notation_path, taala_path)

    assert type(parsed_notation) == annotations.EventData
    assert type(parsed_notation.intervals) is np.ndarray
    assert type(parsed_notation.events) is list

    assert parsed_notation.events[:5] == ['R', ',', 'G', ',', 'G']
    assert parsed_notation.events[-5:] == ['R', 'S', ',', 'R', 'G']

    assert np.array_equal(
        parsed_notation.intervals[:4],
        np.array(
            [
                [0., 3.1299319727891155],
                [3.1299319727891155, 3.7754648526077097],
                [3.7754648526077097, 4.519818594104309],
                [4.519818594104309, 5.188299319727891],
            ]
        ),
    )


def test_load_sections():
    data_home = "tests/resources/mir_datasets/compmusic_carnatic_varnam"
    track = compmusic_carnatic_varnam.Track("dharini_abhogi", data_home=data_home)
    notation_path = track.notation_path
    taala_path = track.taala_path
    parsed_sections = compmusic_carnatic_varnam.load_sections(notation_path, taala_path)

    # Check types
    assert type(parsed_sections) == annotations.SectionData
    assert type(parsed_sections.intervals) is np.ndarray
    assert type(parsed_sections.labels) is list

    assert parsed_sections.intervals[0, 0] == 27.148526077097507
    assert parsed_sections.intervals[0, 1] == 105.75151927437642
    assert parsed_sections.labels == ['pallavi', 'anupallavi', 'muktayiswaram', 'charanam', 'chittiswaram']

    assert compmusic_carnatic_varnam.load_sections(notation_path, None) is None
    assert compmusic_carnatic_varnam.load_sections(None, taala_path) is None
    assert compmusic_carnatic_varnam.load_sections(None, None) is None


def test_load_audio():
    data_home = "tests/resources/mir_datasets/compmusic_carnatic_varnam"
    track = compmusic_carnatic_varnam.Track("dharini_abhogi", data_home=data_home)
    audio_path = track.audio_path
    audio, sr = compmusic_carnatic_varnam.load_audio(audio_path)

    assert sr == 22100
    assert type(audio) == np.ndarray

    assert compmusic_carnatic_varnam.load_audio(None) is None
