from tests.test_utils import run_track_tests

from mirdata.datasets import compmusic_indian_tonic


def test_track():
    default_trackid = "0a6ebaa4-87cc-452d-a7af-a2006e96f16a_0-180"
    data_home = "tests/resources/mir_datasets/compmusic_indian_tonic"
    dataset = compmusic_indian_tonic.Dataset(data_home)
    track = dataset.track(default_trackid)
    expected_attributes = {
        "audio_path": "tests/resources/mir_datasets/compmusic_indian_tonic/indian_art_music_tonic_1.0/"
        + "CM/audio/0a6ebaa4-87cc-452d-a7af-a2006e96f16a_0-180.mp3",
        "track_id": "0a6ebaa4-87cc-452d-a7af-a2006e96f16a_0-180",
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
            "tradition": str,
        },
    )

    _, sr = track.audio
    assert sr == 44100


def test_to_jams():
    default_trackid = "0a6ebaa4-87cc-452d-a7af-a2006e96f16a_0-180"
    data_home = "tests/resources/mir_datasets/compmusic_indian_tonic"
    dataset = compmusic_indian_tonic.Dataset(data_home)
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    assert jam.validate()

    # Test annotations
    assert jam.sandbox.tonic == 131.436
    assert jam.file_metadata.artist == "T. N. Seshagopalan"
