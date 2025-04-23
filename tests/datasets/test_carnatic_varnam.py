# -*- coding: utf-8 -*-
import pytest
import numpy as np
from mirdata import annotations
from mirdata.datasets import compmusic_carnatic_varnam
from tests.test_utils import run_track_tests
import os


def test_track():
    default_trackid = "dharini_abhogi"
    data_home = os.path.normpath(
        "tests/resources/mir_datasets/compmusic_carnatic_varnam"
    )
    dataset = compmusic_carnatic_varnam.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "dharini_abhogi",
        "audio_path": os.path.join(
            os.path.normpath(
                "tests/resources/mir_datasets/compmusic_carnatic_varnam/carnatic_varnam_1.1/"
            ),
            os.path.normpath(
                "Audio/223578__gopalkoduri__carnatic-varnam-by-dharini-in-abhogi-raaga.mp3"
            ),
        ),
        "taala_path": os.path.join(
            os.path.normpath(
                "tests/resources/mir_datasets/compmusic_carnatic_varnam/carnatic_varnam_1.1/"
            ),
            os.path.normpath(
                "Notations_Annotations/annotations/taalas/abhogi/dharini.svl"
            ),
        ),
        "notation_path": os.path.join(
            os.path.normpath(
                "tests/resources/mir_datasets/compmusic_carnatic_varnam/carnatic_varnam_1.1/"
            ),
            os.path.normpath("Notations_Annotations/notations/abhogi.yaml"),
        ),
        "structure_path": os.path.join(
            os.path.normpath(
                "tests/resources/mir_datasets/compmusic_carnatic_varnam/carnatic_varnam_1.1/"
            ),
            os.path.normpath("Notations_Annotations/notations/abhogi/dharini.yaml"),
        ),
        "artist": "dharini",
        "raaga": "abhogi",
    }

    expected_property_types = {
        "audio": tuple,
        "taala": annotations.BeatData,
        "notation": annotations.EventData,
        "sections": annotations.SectionData,
        "mbid": str,
        "arohanam": list,
        "avarohanam": list,
        "tonic": float,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    _, sr = track.audio
    assert sr == 44100


def test_load_metadata():
    default_trackid = "dharini_abhogi"
    data_home = os.path.normpath(
        "tests/resources/mir_datasets/compmusic_carnatic_varnam"
    )
    dataset = compmusic_carnatic_varnam.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)
    parsed_tonic = track.tonic
    assert parsed_tonic == 200.58


def test_load_taala():
    default_trackid = "dharini_abhogi"
    data_home = os.path.normpath(
        "tests/resources/mir_datasets/compmusic_carnatic_varnam"
    )
    dataset = compmusic_carnatic_varnam.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)
    taala_path = track.taala_path
    parsed_taala = compmusic_carnatic_varnam.load_taala(taala_path)

    # Check types
    assert type(parsed_taala) == annotations.BeatData
    assert type(parsed_taala.times) is np.ndarray
    assert type(parsed_taala.positions) is np.ndarray

    # Check values
    assert np.array_equal(
        parsed_taala.times[:4],
        np.array(
            [
                3.1299319727891155,
                3.7754648526077097,
                4.519818594104309,
                5.188299319727891,
            ]
        ),
    )
    assert np.array_equal(parsed_taala.positions[:4], np.array([0, 0, 0, 0]))
    assert compmusic_carnatic_varnam.load_taala(None) is None


def test_load_notation():
    default_trackid = "dharini_abhogi"
    data_home = os.path.normpath(
        "tests/resources/mir_datasets/compmusic_carnatic_varnam"
    )
    dataset = compmusic_carnatic_varnam.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)
    notation_path = track.notation_path
    taala_path = track.taala_path
    structure_path = track.structure_path
    parsed_data = compmusic_carnatic_varnam.load_notation(
        notation_path, taala_path, structure_path
    )
    with pytest.raises(FileNotFoundError):
        parsed_data = compmusic_carnatic_varnam.load_notation(
            "a/fake/path", taala_path, structure_path
        )
    with pytest.raises(FileNotFoundError):
        parsed_data = compmusic_carnatic_varnam.load_notation(
            notation_path, "a/fake/path", structure_path
        )
    with pytest.raises(FileNotFoundError):
        parsed_data = compmusic_carnatic_varnam.load_notation(
            notation_path, taala_path, "a/fake/path"
        )
    parsed_notation = parsed_data[0]
    parsed_sections = parsed_data[1]

    assert type(parsed_notation) == annotations.EventData
    assert type(parsed_notation.intervals) is np.ndarray
    assert type(parsed_notation.events) is list

    assert parsed_notation.events[:5] == ["R,", "G,", "GR", "S,", "SR"]
    assert parsed_notation.events[-5:] == ["GM", "GR", "SR", "GS", "RG"]

    assert np.array_equal(
        parsed_notation.intervals[:3],
        np.array(
            [
                [3.1299319727891155, 3.7754648526077097],
                [3.7754648526077097, 4.519818594104309],
                [4.519818594104309, 5.188299319727891],
            ]
        ),
    )

    # Check types
    assert type(parsed_sections) == annotations.SectionData
    assert type(parsed_sections.intervals) is np.ndarray
    assert type(parsed_sections.labels) is list

    assert parsed_sections.intervals[0, 0] == 3.1299319727891155
    assert parsed_sections.intervals[0, 1] == 48.81401360544218
    assert parsed_sections.labels == [
        "pallavi",
        "anupallavi",
        "muktayiswaram",
        "pallavi",
        "anupallavi",
        "muktayiswaram",
        "charanam",
        "firstchittiswaram",
        "charanam",
        "firstchittiswaram",
        "charanam",
        "secondchittiswaram",
        "charanam",
        "secondchittiswaram",
        "charanam",
        "thirdchittiswaram",
        "charanam",
        "thirdchittiswaram",
        "charanam",
        "fourthchittiswaram",
        "charanam",
        "charanam",
        "fourthchittiswaram",
        "charanam",
    ]


def test_load_mbid():
    default_trackid = "dharini_abhogi"
    data_home = os.path.normpath(
        "tests/resources/mir_datasets/compmusic_carnatic_varnam"
    )
    dataset = compmusic_carnatic_varnam.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)
    notation_path = track.notation_path
    parsed_mbid = compmusic_carnatic_varnam.load_mbid(notation_path)

    assert type(parsed_mbid) == str
    assert parsed_mbid == "6ef7a09c-e08d-46a4-b8bf-891d20e87457"


def test_load_moorchanas():
    default_trackid = "dharini_abhogi"
    data_home = os.path.normpath(
        "tests/resources/mir_datasets/compmusic_carnatic_varnam"
    )
    dataset = compmusic_carnatic_varnam.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)
    notation_path = track.notation_path
    parsed_moorchanas = compmusic_carnatic_varnam.load_moorchanas(notation_path)

    assert type(parsed_moorchanas) == list
    assert type(parsed_moorchanas[0]) == list
    assert type(parsed_moorchanas[0][0]) == str
    assert parsed_moorchanas[0] == ["S", "R2", "G1", "M1", "D2", "S^"]
    assert parsed_moorchanas[1] == ["S^", "D2", "M1", "G1", "R2", "S"]


def test_load_audio():
    default_trackid = "dharini_abhogi"
    data_home = os.path.normpath(
        "tests/resources/mir_datasets/compmusic_carnatic_varnam"
    )
    dataset = compmusic_carnatic_varnam.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)
    audio_path = track.audio_path
    audio, sr = compmusic_carnatic_varnam.load_audio(audio_path)

    assert sr == 44100
    assert type(audio) == np.ndarray

    assert compmusic_carnatic_varnam.load_audio(None) is None
