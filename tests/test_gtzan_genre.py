import os

from tests.test_utils import run_track_tests

from mirdata.datasets import gtzan_genre

TEST_DATA_HOME = "tests/resources/mir_datasets/gtzan_genre"


def test_track():
    default_trackid = "country.00000"
    dataset = gtzan_genre.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    expected_attributes = {
        "genre": "country",
        "audio_path": "tests/resources/mir_datasets/gtzan_genre/"
        + "gtzan_genre/genres/country/country.00000.wav",
        "track_id": "country.00000",
    }
    expected_properties = {"audio": tuple}
    run_track_tests(track, expected_attributes, expected_properties)

    audio, sr = track.audio
    assert sr == 22050
    assert audio.shape == (663300,)


def test_hiphop():
    dataset = gtzan_genre.Dataset(TEST_DATA_HOME)
    track = dataset.track("hiphop.00000")
    assert track.genre == "hip-hop"


def test_to_jams():
    default_trackid = "country.00000"
    dataset = gtzan_genre.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    # Validate GTZAN schema
    assert jam.validate()

    # Test the that the genre parser of mirdata is correct
    assert jam.annotations["tag_gtzan"][0].data[0].value == "country"
