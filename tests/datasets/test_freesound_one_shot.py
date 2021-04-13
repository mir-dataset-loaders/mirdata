import numpy as np

from tests.test_utils import run_track_tests
from mirdata.datasets import freesound_one_shot_percussive_sounds

TEST_DATA_HOME = "tests/resources/mir_datasets/freesound_one_shot_percussive_sounds"


def test_track():
    default_trackid = "183"
    dataset = freesound_one_shot_percussive_sounds.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "audio_path": "tests/resources/mir_datasets/freesound_one_shot_percussive_sounds/"
        + "one_shot_percussive_sounds/1/183.wav",
        "analysis_path": "tests/resources/mir_datasets/freesound_one_shot_percussive_sounds/"
        + "analysis/1/183_analysis.json",
        "track_id": "183",
    }

    expected_property_types = {
        "audio": tuple,
        "analysis": dict,
        "filename": str,
        "username": str,
        "license": str,
    }

    run_track_tests(track, expected_attributes, expected_property_types)


def test_to_jams():
    default_trackid = "183"
    dataset = freesound_one_shot_percussive_sounds.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    # Validate cante100 jam schema
    assert jam.validate()

    assert jam.file_metadata["duration"] == 0.34575000405311584
    assert jam.annotations == []

    # Check jam sandbox
    jam_metadata = jam.sandbox
    assert jam_metadata["lossless"] == 1.0
    assert jam_metadata["codec"] == "pcm_s16le"
    assert jam_metadata["bitrate"] == 256000.0
    assert jam_metadata["samplerate"] == 16000.0
    assert jam_metadata["channels"] == 1.0
    assert jam_metadata["audio_md5"] == "dd11a896b4d08c2a93d6480bb3e40016"
    assert jam_metadata["loudness"] == -18.421449661254883
    assert jam_metadata["dynamic_range"] == 0.0
    assert jam_metadata["temporal_centroid"] == 0.4129089415073395
    assert jam_metadata["log_attack_time"] == -1.4238076210021973
    assert jam_metadata["filesize"] == 11108
    assert jam_metadata["single_event"] is True
    assert jam_metadata["hardness"] == 63.75773273351933
    assert jam_metadata["depth"] == 44.40461964558294
    assert jam_metadata["brightness"] == 72.8391772655705
    assert jam_metadata["roughness"] == 69.52154422554737
    assert jam_metadata["warmth"] == 35.066030998902875
    assert jam_metadata["sharpness"] == 61.042021710446086
    assert jam_metadata["boominess"] == 23.631757376023458
    assert jam_metadata["reverb"] is False


def test_load_analysis():
    default_trackid = "183"
    dataset = freesound_one_shot_percussive_sounds.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    analysis_path = track.analysis_path
    analysis = freesound_one_shot_percussive_sounds.load_analysis(analysis_path)

    # check analysis elements
    assert type(analysis) is dict
    assert analysis.get("duration") == 0.34575000405311584
    assert analysis.get("lossless") == 1.0
    assert analysis.get("codec") == "pcm_s16le"
    assert analysis.get("bitrate") == 256000.0
    assert analysis.get("samplerate") == 16000.0
    assert analysis.get("channels") == 1.0
    assert analysis.get("audio_md5") == "dd11a896b4d08c2a93d6480bb3e40016"
    assert analysis.get("loudness") == -18.421449661254883
    assert analysis.get("dynamic_range") == 0.0
    assert analysis.get("temporal_centroid") == 0.4129089415073395
    assert analysis.get("log_attack_time") == -1.4238076210021973
    assert analysis.get("filesize") == 11108
    assert analysis.get("single_event") is True
    assert analysis.get("hardness") == 63.75773273351933
    assert analysis.get("depth") == 44.40461964558294
    assert analysis.get("brightness") == 72.8391772655705
    assert analysis.get("roughness") == 69.52154422554737
    assert analysis.get("warmth") == 35.066030998902875
    assert analysis.get("sharpness") == 61.042021710446086
    assert analysis.get("boominess") == 23.631757376023458
    assert analysis.get("reverb") is False


def test_load_audio():
    default_trackid = "183"
    dataset = freesound_one_shot_percussive_sounds.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    audio_path = track.audio_path
    audio, sr = freesound_one_shot_percussive_sounds.load_audio(audio_path)
    assert sr == 16000
    assert type(audio) is np.ndarray


def test_metadata():
    default_trackid = "183"
    dataset = freesound_one_shot_percussive_sounds.Dataset(TEST_DATA_HOME)
    metadata = dataset._metadata
    print(metadata[default_trackid])
    assert type(metadata) is dict
    assert metadata[default_trackid].get("name") == "1.wav"
    assert metadata[default_trackid].get("username") == "plagasul"
    assert (
        metadata[default_trackid].get("license")
        == "http://creativecommons.org/licenses/by/3.0/"
    )
