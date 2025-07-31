import os
import pytest
import shutil
import xml.etree.ElementTree as ET

from mirdata.datasets import idmt_smt_audio_effects
from mirdata import download_utils
from tests.test_utils import run_track_tests


TEST_DATA_HOME = os.path.normpath("tests/resources/mir_datasets/idmt_smt_audio_effects")


def test_track():
    default_trackid = "G73-45200-3341-33944"
    dataset = idmt_smt_audio_effects.Dataset(TEST_DATA_HOME, version="test")
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
        "audio": [
            "Gitarre monophon2/Samples/Tremolo/G73-45200-3341-33944.wav",
            "4b8c1e95cc99cd1ecd83f46ee9b604ba",
        ]
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 44100
    assert audio.shape == (88201,)


def test_metadata():
    dataset = idmt_smt_audio_effects.Dataset(TEST_DATA_HOME, version="test")
    metadata = dataset._metadata
    track_metadata = metadata["G73-45200-3341-33944"]
    assert track_metadata["fx_group"] == 3
    assert track_metadata["fx_setting"] == 1
    assert track_metadata["instrument"] == "G"
    assert track_metadata["midi_nr"] == 45

    fake_data_home = os.path.join(TEST_DATA_HOME, "fake_directory")

    # Ensure the directory exists (but it should not have any XML inside!)
    if not os.path.exists(fake_data_home):
        os.makedirs(fake_data_home)

    # Test for FileNotFoundError
    with pytest.raises(FileNotFoundError):
        dataset = idmt_smt_audio_effects.Dataset(fake_data_home, version="test")
        if hasattr(dataset, "_metadata"):
            del dataset._metadata
        metadata = dataset._metadata

    # Test for ParseError
    corrupted_data_home = os.path.join(TEST_DATA_HOME)

    # Create a fake corrupted XML file
    with open(os.path.join(corrupted_data_home, "corrupted.xml"), "w") as f:
        f.write("<root><corrupted>")

    dataset = idmt_smt_audio_effects.Dataset(corrupted_data_home, version="test")
    with pytest.raises(ValueError):
        metadata = dataset._metadata

    # Clean up after the test by removing the corrupted XML
    os.remove(os.path.join(corrupted_data_home, "corrupted.xml"))


def test_download(httpserver):
    data_home = "tests/resources/mir_datasets/idmt_smt_audio_effects_download"
    if os.path.exists(data_home):
        shutil.rmtree(data_home)

    httpserver.serve_content(
        open("tests/resources/download/IDMT-SMT-AUDIO-EFFECTS.zip", "rb").read()
    )

    remotes = {
        "full_dataset": download_utils.RemoteFileMetadata(
            filename="IDMT-SMT-AUDIO-EFFECTS.zip",
            url=httpserver.url,
            checksum=("462b32e4834749204347014c819b3a7c"),
        )
    }

    dataset = idmt_smt_audio_effects.Dataset(data_home, version="test")
    dataset.remotes = remotes
    dataset.download(None, False, False)

    assert os.path.exists(data_home)

    # expected_folders = [
    #     "Bass monophon",
    #     "Bass monophon2",
    #     "Gitarre monophon",
    #     "Gitarre monophon2",
    #     "Gitarre polyphon",
    #     "Gitarre polyphon2",
    # ]

    # test downloading again
    dataset.download(None, False, False)

    if os.path.exists(data_home):
        shutil.rmtree(data_home)

    # test downloading twice with cleanup
    dataset.download(None, False, True)
    dataset.download(None, False, False)

    if os.path.exists(data_home):
        shutil.rmtree(data_home)

    # test downloading twice with force overwrite
    dataset.download(None, False, False)
    dataset.download(None, True, False)

    if os.path.exists(data_home):
        shutil.rmtree(data_home)

    # test downloading twice with force overwrite and cleanup
    dataset.download(None, False, True)
    dataset.download(None, True, False)

    if os.path.exists(data_home):
        shutil.rmtree(data_home)
