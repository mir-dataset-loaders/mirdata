import os

from tests.test_utils import run_track_tests

from mirdata.datasets import idmt_smt_audio_effects


def test_track():
    default_trackid = "G73-45200-3341-33944"
<<<<<<< HEAD
    data_home = os.path.normpath(
        "tests/resources/mir_datasets/idmt_smt_audio_effects"
    )
=======
    data_home = os.path.normpath("tests/resources/mir_datasets/idmt_smt_audio_effects")
>>>>>>> f327b32 (added tests)
    dataset = idmt_smt_audio_effects.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "G73-45200-3341-33944",
        "audio_path": os.path.normpath(
            "tests/resources/mir_datasets/idmt_smt_audio_effects/Gitarre monophon2/Samples/Tremolo/G73-45200-3341-33944.wav",
        ),
    }

    expected_property_types = {
        "audio": tuple,
        "fx_group": int,
        "fx_setting": int,
        "fx_type": int,
        "instrument": str,
        "midi_nr": int,
    }

    assert track._track_paths == {
<<<<<<< HEAD
        "audio": ["Gitarre monophon2/Samples/Tremolo/G73-45200-3341-33944.wav",
                    "4b8c1e95cc99cd1ecd83f46ee9b604ba"]
=======
        "audio": [
            "Gitarre monophon2/Samples/Tremolo/G73-45200-3341-33944.wav",
            "4b8c1e95cc99cd1ecd83f46ee9b604ba",
        ]
>>>>>>> f327b32 (added tests)
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 44100
    assert audio.shape == (88201,)


def test_to_jams():
    default_trackid = "G73-45200-3341-33944"
    data_home = os.path.normpath("tests/resources/mir_datasets/idmt_smt_audio_effects")
    dataset = idmt_smt_audio_effects.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    assert jam["sandbox"]["fx_group"] == 3
    assert jam["sandbox"]["fx_setting"] == 1
    assert jam["sandbox"]["instrument"] == "G"
    assert jam["sandbox"]["midi_nr"] == 45


def test_metadata():
    data_home = os.path.normpath("tests/resources/mir_datasets/idmt_smt_audio_effects")
    dataset = idmt_smt_audio_effects.Dataset(data_home, version="test")
    metadata = dataset._metadata
    track_metadata = metadata["G73-45200-3341-33944"]
    assert track_metadata["fx_group"] == 3
    assert track_metadata["fx_setting"] == 1
    assert track_metadata["instrument"] == "G"
    assert track_metadata["midi_nr"] == 45
