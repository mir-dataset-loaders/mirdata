import numpy as np
from mirdata import annotations
from mirdata.datasets import compmusic_carnatic_rhythm
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "1-04_Shri_Visvanatham"
    data_home = "tests/resources/mir_datasets/compmusic_carnatic_rhythm"
    dataset = compmusic_carnatic_rhythm.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    print(dataset._metadata.keys())
    print(track._track_metadata)
    print(track.meter)
    print(track.artists)
    print(track.raaga)
    print(track.start_time)


    expected_attributes = {
        "track_id": "1-04_Shri_Visvanatham",
        "audio_path": "tests/resources/mir_datasets/compmusic_carnatic_rhythm/CMR_subset_1.0/"
        + "audio/01_10003_1-04_Shri_Visvanatham.wav",
        "beats_path": "tests/resources/mir_datasets/compmusic_carnatic_rhythm/CMR_subset_1.0/"
        + "annotations/beats/01_10003_1-04_Shri_Visvanatham.beats",
        "meter_path": "tests/resources/mir_datasets/compmusic_carnatic_rhythm/CMR_subset_1.0/"
        + "annotations/meter/01_10003_1-04_Shri_Visvanatham.meter"
    }

    expected_property_types = {
        "meter": str,
        "beats": annotations.BeatData,
        "audio": tuple,
        "mbid": str,
        "name": str,
        "artists": str,
        "release": str,
        "lead_instrument_code": str,
        "taala": str,
        "raaga": str,
        "start_time": None,
        "end_time": None,
        "length_seconds": None,
        "length_minutes": None,
        "num_of_beats": int,
        "num_of_samas": int,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 44100
    assert audio.shape[0] == 2


def test_to_jams():
    data_home = "tests/resources/mir_datasets/saraga_carnatic"
    dataset = compmusic_carnatic_rhythm.Dataset(data_home, version="test")
    track = dataset.track("1-04_Shri_Visvanatham")
    jam = track.to_jams()

    # Tonic
    assert jam["sandbox"].tonic == 201.740890

    # Sama
    samas = jam.search(namespace="beat")[0]["data"]
    assert len(samas) == 3
    assert [sama.time for sama in samas] == [4.894, 10.229, 15.724]
    assert [sama.duration for sama in samas] == [0.0, 0.0, 0.0]
    assert [sama.value for sama in samas] == [1, 2, 3]
    assert [sama.confidence for sama in samas] == [None, None, None]


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
    assert metadata["concert"] == [
        {
            "mbid": "0816586d-c83e-4c79-a0aa-9b0e578f408d",
            "title": "Cherthala Ranganatha Sharma at Arkay",
        }
    ]


def test_load_meter():
    data_home = "tests/resources/mir_datasets/saraga_carnatic"
    dataset = compmusic_carnatic_rhythm.Dataset(data_home, version="test")
    track = dataset.track("1-04_Shri_Visvanatham")
    tonic_path = track.ctonic_path
    parsed_tonic = compmusic_carnatic_rhythm.load_meter(tonic_path)
    assert parsed_tonic == 201.740890
    assert compmusic_carnatic_rhythm.load_meter(None) is None


def test_load_sama():
    data_home = "tests/resources/mir_datasets/compmusic_carnatic_rhythm"
    dataset = compmusic_carnatic_rhythm.Dataset(data_home, version="test")
    track = dataset.track("1-04_Shri_Visvanatham")
    sama_path = track.sama_path
    parsed_sama = compmusic_carnatic_rhythm.load_beats(sama_path)

    # Check types
    assert type(parsed_sama) == annotations.BeatData
    assert type(parsed_sama.times) is np.ndarray
    assert type(parsed_sama.positions) is np.ndarray

    # Check values
    assert np.array_equal(parsed_sama.times, np.array([4.894, 10.229, 15.724]))
    assert np.array_equal(parsed_sama.positions, np.array([1, 2, 3]))
    assert compmusic_carnatic_rhythm.load_sama(None) is None



def test_load_metadata():
    data_home = "tests/resources/mir_datasets/saraga_carnatic"
    dataset = compmusic_carnatic_rhythm.Dataset(data_home, version="test")
    # get dataset metadata

    assert parsed_metadata["raaga"] == [
        {
            "uuid": "42dd0ccb-f92a-4622-ae5d-a3be571b4939",
            "name": "Śrīranjani",
            "common_name": "shri ranjani",
        }
    ]
    assert parsed_metadata["form"] == [{"name": "Kriti"}]
    assert parsed_metadata["title"] == "Bhuvini Dasudane"
    assert parsed_metadata["work"] == [
        {"mbid": "4d05ce9b-c45e-4c85-9eca-941d68b61132", "title": "Bhuvini Dasudane"}
    ]
    assert parsed_metadata["length"] == 309000
    assert parsed_metadata["taala"] == [
        {
            "uuid": "c788c38a-b53a-48cb-b7bf-d11769260c4d",
            "name": "Ādi",
            "common_name": "adi",
        }
    ]
    assert parsed_metadata["album_artists"] == [
        {
            "mbid": "e09b0542-84e1-45ad-b09a-a05a9ad0cb83",
            "name": "Cherthala Ranganatha Sharma",
        }
    ]
    assert parsed_metadata["mbid"] == "9f5a5452-14cb-4af0-9289-4833854ee60d"
    assert parsed_metadata["concert"] == [
        {
            "mbid": "0816586d-c83e-4c79-a0aa-9b0e578f408d",
            "title": "Cherthala Ranganatha Sharma at Arkay",
        }
    ]


def test_load_audio():
    data_home = "tests/resources/mir_datasets/compmusic_carnatic_rhythm"
    dataset = compmusic_carnatic_rhythm.Dataset(data_home, version="test")
    track = dataset.track("1-04_Shri_Visvanatham")
    audio_path = track.audio_path
    audio, sr = compmusic_carnatic_rhythm.load_audio(audio_path)

    assert sr == 44100
    assert type(audio) == np.ndarray
    assert audio.shape[0] == 2

    assert compmusic_carnatic_rhythm.load_audio(None) is None
