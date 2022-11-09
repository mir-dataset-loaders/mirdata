import os

from tests.test_utils import run_track_tests

from mirdata.datasets import compmusic_indian_tonic


def test_track():
    default_trackid = "402f49e2-5957-4b24-9229-0c94b0c4c07d_0-180"
    data_home = "tests/resources/mir_datasets/compmusic_indian_tonic"
    dataset = compmusic_indian_tonic.Dataset(data_home)
    track = dataset.track(default_trackid)
    expected_attributes = {
        "audio_path": "tests/resources/mir_datasets/compmusic_indian_tonic/indian_art_music_tonic_1.0/"
        + "CM/audio/402f49e2-5957-4b24-9229-0c94b0c4c07d_0-180.mp3",
        "track_id": "402f49e2-5957-4b24-9229-0c94b0c4c07d_0-180",
    }

    run_track_tests(
        track,
        expected_attributes,
        {
            "audio": tuple,
            "tonic": float,
            "artist": str,
            "gender": str,
            "mbid": str,
            "type": str,
            "tradition": str
        }
    )

    _, sr = track.audio
    assert sr == 44100


def test_to_jams():
    default_trackid = "402f49e2-5957-4b24-9229-0c94b0c4c07d_0-180"
    data_home = "tests/resources/mir_datasets/compmusic_indian_tonic"
    dataset = compmusic_indian_tonic.Dataset(data_home)
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    assert jam.validate()

    # Test annotations
    assert jam.sandbox.tonic == 
    parsed_artists = jam.sandbox.artist
    parsed_gender = jam.sandbox.gender
    parsed_mbid = jam.sandbox.mbid
    parsed_type = jam.sandbox.type
    parsed_tradition = jam.sandbox.tradition


