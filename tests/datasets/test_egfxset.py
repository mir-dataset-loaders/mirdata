from mirdata.datasets import egfxset
from tests.test_utils import run_track_tests


def test_track():
    data_home = "tests/resources/mir_datasets/egfxset"
    dataset = egfxset.Dataset(data_home, version="test")

    track = dataset.track("TapeEcho_Bridge/2-0")

    expected_attributes = {
        "track_id": "TapeEcho_Bridge/2-0",
        "audio_path": "tests/resources/mir_datasets/egfxset/TapeEcho/Bridge/2-0.wav",
        "stringfret_tuple": "2-0",
        "note": "B3",
        "midinote": 59,
        "pickup_configuration": "Bridge",
        "effect": "tape echo",
        "model": "Line 6 DL4 Delay",
        "effect_type": "delay",
        "knob_names": [
            "effect selector",
            "delay time",
            "repeats",
            "tweak (bass)",
            "tweez (treble)",
            "mix",
        ],
        "knob_type": ["selector", "rate", "effect decay", "eq", "eq", "effect amount"],
        "setting": ["tape echo", "120 bpm", 0.6, 0.5, 0.5, 0.5],
    }

    expected_property_types = {
        "audio": tuple,
    }

    assert track._track_paths == {
        "audio": ["TapeEcho/Bridge/2-0.wav", "bf9041e98fbc3c1145583d1601ab2d7b"]
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 48000
    assert audio.shape == (48000 * 5,)

    track = dataset.track("Clean_Middle/6-22")

    expected_attributes = {
        "track_id": "Clean_Middle/6-22",
        "audio_path": "tests/resources/mir_datasets/egfxset/Clean/Middle/6-22.wav",
        "stringfret_tuple": "6-22",
        "note": "D4",
        "midinote": 62,
        "pickup_configuration": "Middle",
        "effect": "clean",
        "model": "None",
        "effect_type": "None",
        "knob_names": "None",
        "knob_type": "None",
        "setting": "None",
    }

    expected_property_types = {
        "audio": tuple,
    }

    assert track._track_paths == {
        "audio": ["Clean/Middle/6-22.wav", "93c580d88d65400804f5c8f88f715ec1"]
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 48000
    assert audio.shape == (48000 * 5,)


def test_to_jams():

    data_home = "tests/resources/mir_datasets/egfxset"
    dataset = egfxset.Dataset(data_home, version="test")

    # Case with a TapeEcho track
    track = dataset.track("TapeEcho_Bridge/2-0")
    jam = track.to_jams()

    assert jam["sandbox"]["String-fret Tuple"] == "2-0"
    assert jam["sandbox"]["Note"] == "B3"
    assert jam["sandbox"]["Midinote"] == 59
    assert jam["sandbox"]["Pickup Configuration"] == "Bridge"
    assert jam["sandbox"]["Effect"] == "tape echo"
    assert jam["sandbox"]["Model"] == "Line 6 DL4 Delay"
    assert jam["sandbox"]["Effect Type"] == "delay"
    assert jam["sandbox"]["Knob Names"] == [
        "effect selector",
        "delay time",
        "repeats",
        "tweak (bass)",
        "tweez (treble)",
        "mix",
    ]
    assert jam["sandbox"]["Knob Type"] == [
        "selector",
        "rate",
        "effect decay",
        "eq",
        "eq",
        "effect amount",
    ]
    assert jam["sandbox"]["Setting"] == ["tape echo", "120 bpm", 0.6, 0.5, 0.5, 0.5]

    # Case with a Clean track
    track = dataset.track("Clean_Middle/6-22")
    jam = track.to_jams()

    assert jam["sandbox"]["String-fret Tuple"] == "6-22"
    assert jam["sandbox"]["Note"] == "D4"
    assert jam["sandbox"]["Midinote"] == 62
    assert jam["sandbox"]["Pickup Configuration"] == "Middle"
    assert jam["sandbox"]["Effect"] == "clean"
    assert jam["sandbox"]["Model"] == "None"
    assert jam["sandbox"]["Effect Type"] == "None"
    assert jam["sandbox"]["Knob Names"] == "None"
    assert jam["sandbox"]["Knob Type"] == "None"
    assert jam["sandbox"]["Setting"] == "None"
