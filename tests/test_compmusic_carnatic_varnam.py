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
        "notation": str,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 22100


def test_to_jams():
    data_home = "tests/resources/mir_datasets/saraga_carnatic"
    track = saraga_carnatic.Track("116_Bhuvini_Dasudane", data_home=data_home)
    jam = track.to_jams()

    # Tonic
    assert jam["sandbox"].tonic == 201.740890

    # Pitch
    pitches = jam.search(namespace="pitch_contour")[0]["data"]
    assert len(pitches) == 6
    assert [pitch.time for pitch in pitches] == [
        0.0000000,
        0.0044444,
        0.0088889,
        0.0133333,
        0.0177778,
        0.0222222,
    ]
    assert [pitch.duration for pitch in pitches] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert [pitch.value for pitch in pitches] == [
        {"index": 0, "frequency": 0.0000000, "voiced": False},
        {"index": 0, "frequency": 100.1200000, "voiced": True},
        {"index": 0, "frequency": 200.2300000, "voiced": True},
        {"index": 0, "frequency": 300.3400000, "voiced": True},
        {"index": 0, "frequency": 400.4300000, "voiced": True},
        {"index": 0, "frequency": 600.12300000, "voiced": True},
    ]
    assert [pitch.confidence for pitch in pitches] == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    pitches_vocal = jam.search(namespace="pitch_contour")[1]["data"]
    assert len(pitches_vocal) == 6
    assert [pitch_vocal.time for pitch_vocal in pitches_vocal] == [
        0.000000000000000000e00,
        2.902494331065759697e-03,
        5.804988662131519393e-03,
        8.707482993197278656e-03,
        1.160997732426303879e-02,
        1.451247165532879892e-02,
    ]
    assert [pitch_vocal.duration for pitch_vocal in pitches_vocal] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert [pitch_vocal.value for pitch_vocal in pitches_vocal] == [
        {"index": 0, "frequency": 0.000000000000000000e00, "voiced": False},
        {"index": 0, "frequency": 1.123456789012345678e02, "voiced": True},
        {"index": 0, "frequency": 2.234567890123456789e02, "voiced": True},
        {"index": 0, "frequency": 3.345678901234567890e02, "voiced": True},
        {"index": 0, "frequency": 4.456789012345678901e01, "voiced": True},
        {"index": 0, "frequency": 0.000000000000000000e00, "voiced": False},
    ]
    assert [pitch_vocal.confidence for pitch_vocal in pitches_vocal] == [
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
    ]

    # Tempo
    parsed_tempo = jam["sandbox"].tempo
    assert parsed_tempo == {
        "tempo_apm": 330,
        "tempo_bpm": 82,
        "sama_interval": 5.827,
        "beats_per_cycle": 32,
        "subdivisions": 4,
    }

    # Sama
    samas = jam.search(namespace="beat")[0]["data"]
    assert len(samas) == 3
    assert [sama.time for sama in samas] == [4.894, 10.229, 15.724]
    assert [sama.duration for sama in samas] == [0.0, 0.0, 0.0]
    assert [sama.value for sama in samas] == [1, 1, 1]
    assert [sama.confidence for sama in samas] == [None, None, None]

    # Sections
    sections = jam.search(namespace="segment_open")[0]["data"]
    assert [section.time for section in sections] == [
        0.065306122,
        85.35510204,
        167.314285714,
    ]
    assert [section.duration for section in sections] == [
        85.289795918,
        81.95918367399999,
        142.02775510200001,
    ]
    assert [section.value for section in sections] == [
        "Pallavi",
        "Anupallavi",
        "Caraṇam",
    ]
    assert [section.confidence for section in sections] == [None, None, None]

    # Phrases
    phrases = jam.search(namespace="tag_open")[0]["data"]
    assert [phrase.time for phrase in phrases] == [0.224489795, 5.844897959, 8.50430839]
    assert [phrase.duration for phrase in phrases] == [
        2.4938775509999997,
        2.4734693870000006,
        2.2755555550000004,
    ]
    assert [phrase.value for phrase in phrases] == [
        "ndmdnsndn",
        "ndmdnsndn",
        "ndmdndmgr",
    ]
    assert [phrase.confidence for phrase in phrases] == [None, None, None]

    # Metadata
    metadata = jam["sandbox"].metadata
    assert metadata["raaga"] == [
        {
            "uuid": "42dd0ccb-f92a-4622-ae5d-a3be571b4939",
            "name": "Śrīranjani",
            "common_name": "shri ranjani",
        }
    ]
    assert metadata["form"] == [{"name": "Kriti"}]
    assert metadata["title"] == "Bhuvini Dasudane"
    assert metadata["work"] == [
        {"mbid": "4d05ce9b-c45e-4c85-9eca-941d68b61132", "title": "Bhuvini Dasudane"}
    ]
    assert metadata["length"] == 309000
    assert metadata["taala"] == [
        {
            "uuid": "c788c38a-b53a-48cb-b7bf-d11769260c4d",
            "name": "Ādi",
            "common_name": "adi",
        }
    ]
    assert metadata["album_artists"] == [
        {
            "mbid": "e09b0542-84e1-45ad-b09a-a05a9ad0cb83",
            "name": "Cherthala Ranganatha Sharma",
        }
    ]
    assert metadata["mbid"] == "9f5a5452-14cb-4af0-9289-4833854ee60d"
    assert metadata["artists"] == [
        {
            "instrument": {
                "mbid": "c5aa7d98-c14d-4ff1-8afb-f8743c62496c",
                "name": "Ghatam",
            },
            "attributes": "",
            "lead": False,
            "artist": {
                "mbid": "19f93366-5d58-47f1-bc4f-9225ac7af6ba",
                "name": "N Guruprasad",
            },
        },
        {
            "instrument": {
                "mbid": "f689271c-37bc-4c49-92a3-a14b15ee5d0e",
                "name": "Mridangam",
            },
            "attributes": "",
            "lead": False,
            "artist": {
                "mbid": "39c1d741-6154-418b-bf4b-12c77ba13873",
                "name": "Srimushnam V Raja Rao",
            },
        },
        {
            "instrument": {
                "mbid": "089f123c-0f7d-4105-a64e-49de81ca8fa4",
                "name": "Violin",
            },
            "attributes": "",
            "lead": False,
            "artist": {
                "mbid": "a2df55e3-d141-4767-862e-77adca691d4b",
                "name": "B.U. Ganesh Prasad",
            },
        },
        {
            "instrument": {
                "mbid": "d92884b7-ee0c-46d5-96f3-918196ba8c5b",
                "name": "Voice",
            },
            "attributes": "lead vocals",
            "lead": True,
            "artist": {
                "mbid": "e09b0542-84e1-45ad-b09a-a05a9ad0cb83",
                "name": "Cherthala Ranganatha Sharma",
            },
        },
    ]
    assert metadata["concert"] == [
        {
            "mbid": "0816586d-c83e-4c79-a0aa-9b0e578f408d",
            "title": "Cherthala Ranganatha Sharma at Arkay",
        }
    ]
    assert metadata["data_home"] == "tests/resources/mir_datasets/saraga_carnatic"


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
    assert np.array_equal(parsed_taala.times, np.array([
        3.1299319727891155, 3.7754648526077097, 4.519818594104309, 5.188299319727891
    ]))
    assert np.array_equal(parsed_taala.positions, np.array([1, 1, 1, 1]))
    assert compmusic_carnatic_varnam.load_taala(None) is None


def test_load_notation():
    data_home = "tests/resources/mir_datasets/compmusic_carnatic_varnam"
    track = compmusic_carnatic_varnam.Track("dharini_abhogi", data_home=data_home)
    notation_path = track.notation_path
    parsed_notation = compmusic_carnatic_varnam.load_notation(notation_path)

    print(parsed_notation)
    assert parsed_notation == 1


def test_load_audio():
    data_home = "tests/resources/mir_datasets/compmusic_carnatic_varnam"
    track = compmusic_carnatic_varnam.Track("dharini_abhogi", data_home=data_home)
    audio_path = track.audio_path
    audio, sr = compmusic_carnatic_varnam.load_audio(audio_path)

    assert sr == 22100
    assert type(audio) == np.ndarray

    assert compmusic_carnatic_varnam.load_audio(None) is None
