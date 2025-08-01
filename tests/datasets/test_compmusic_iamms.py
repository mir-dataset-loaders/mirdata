import os
import numpy as np
from mirdata import annotations
from mirdata.datasets import compmusic_iamms
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "0_ALB_Bhairavi1_70albmgpnganesan"
    data_home = os.path.normpath("tests/resources/mir_datasets/compmusic_iamms")
    dataset = compmusic_iamms.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "0_ALB_Bhairavi1_70albmgpnganesan",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/compmusic_iamms/"),
            "Carnatic/ALB_Bhairavi1_70albmgpnganesan/",
            "ALB_Bhairavi1_70albmgpnganesan.mp3",
        ),
        "sections_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/compmusic_iamms/"),
            "Carnatic/ALB_Bhairavi1_70albmgpnganesan/",
            "ALB_Bhairavi1_70albmgpnganesan.anot",
        ),
        "sections_finetuned_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/compmusic_iamms/"),
            "Carnatic/ALB_Bhairavi1_70albmgpnganesan/",
            "ALB_Bhairavi1_70albmgpnganesan.anotEdit1",
        ),
        "nyas_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/compmusic_iamms/"),
            "Carnatic/ALB_Bhairavi1_70albmgpnganesan/",
            "ALB_Bhairavi1_70albmgpnganesan.flatSegNyas",
        ),
        "pitch_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/compmusic_iamms/"),
            "Carnatic/ALB_Bhairavi1_70albmgpnganesan/",
            "ALB_Bhairavi1_70albmgpnganesan.pitch",
        ),
        "pitch_finetuned_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/compmusic_iamms/"),
            "Carnatic/ALB_Bhairavi1_70albmgpnganesan/",
            "ALB_Bhairavi1_70albmgpnganesan.pitchSilIntrpPP",
        ),
        "tonic_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/compmusic_iamms/"),
            "Carnatic/ALB_Bhairavi1_70albmgpnganesan/",
            "ALB_Bhairavi1_70albmgpnganesan.tonic",
        ),
        "tonic_finetuned_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/compmusic_iamms/"),
            "Carnatic/ALB_Bhairavi1_70albmgpnganesan/",
            "ALB_Bhairavi1_70albmgpnganesan.tonicFine",
        )
    }

    expected_property_types = {
        "sections": annotations.SectionData,
        "sections_finetuned": annotations.SectionData,
        "pitch": annotations.F0Data,
        "pitch_finetuned": annotations.F0Data,
        "nyas": annotations.EventData,
        "tonic": float,
        "tonic_finetuned": float,
        "audio": tuple
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 44100
    assert audio.shape[0] == 2


def test_load_tonic():
    data_home = "tests/resources/mir_datasets/compmusic_iamms"
    dataset = compmusic_iamms.Dataset(data_home, version="test")
    track = dataset.track("0_ALB_Bhairavi1_70albmgpnganesan")
    tonic_path = track.tonic_path
    tonic_finetuned_path = track.tonic_finetuned_path
    parsed_tonic = compmusic_iamms.load_tonic(tonic_path)
    parsed_tonic_finetuned = compmusic_iamms.load_tonic(tonic_finetuned_path)
    assert parsed_tonic == 134.000
    assert parsed_tonic_finetuned == 134.000
    assert compmusic_iamms.load_tonic(None) is None


def test_load_pitch():
    data_home = "tests/resources/mir_datasets/compmusic_iamms"
    dataset = compmusic_iamms.Dataset(data_home, version="test")
    track = dataset.track("0_ALB_Bhairavi1_70albmgpnganesan")
    pitch_path = track.pitch_path
    pitch_finetuned_path = track.pitch_finetuned_path
    parsed_pitch = compmusic_iamms.load_pitch(pitch_path)
    parsed_pitch_finetuned = compmusic_iamms.load_pitch(pitch_finetuned_path)

    # Check types
    assert type(parsed_pitch) == annotations.F0Data
    assert type(parsed_pitch.times) is np.ndarray
    assert type(parsed_pitch.frequencies) is np.ndarray
    assert type(parsed_pitch.voicing) is np.ndarray
    assert type(parsed_pitch_finetuned) == annotations.F0Data
    assert type(parsed_pitch_finetuned.times) is np.ndarray
    assert type(parsed_pitch_finetuned.frequencies) is np.ndarray
    assert type(parsed_pitch_finetuned.voicing) is np.ndarray

    # Check values
    assert np.array_equal(
        parsed_pitch.times,
        np.array([0.000000000000000000e+00, 2.902494331065759697e-03, 5.804988662131519393e-03, 8.707482993197278656e-03]),
    )
    assert np.array_equal(
        parsed_pitch_finetuned.times,
        np.array([0.000000000000000000e+00, 2.902494331065759697e-03, 5.804988662131519393e-03, 8.707482993197278656e-03]),
    )
    assert np.array_equal(
        parsed_pitch.frequencies,
        np.array(
            [
                0.0000000,
                100.1200000,
                200.2300000,
                300.3400000,
            ]
        ),
    )
    assert np.array_equal(
        parsed_pitch_finetuned.frequencies,
        np.array(
            [
                0.0000000,
                100.1200000,
                200.2300000,
                300.3400000,
            ]
        ),
    )
    assert np.array_equal(
        parsed_pitch.voicing, np.array([0.0, 1.0, 1.0, 1.0])
    )
    assert np.array_equal(
        parsed_pitch_finetuned.voicing, np.array([0.0, 1.0, 1.0, 1.0])
    )

    assert compmusic_iamms.load_pitch(None) is None


def test_load_sections():
    data_home = "tests/resources/mir_datasets/compmusic_iamms"
    dataset = compmusic_iamms.Dataset(data_home, version="test")
    track = dataset.track("0_ALB_Bhairavi1_70albmgpnganesan")
    sections_path = track.sections_path
    sections_finetuned_path = track.sections_finetuned_path
    parsed_sections = compmusic_iamms.load_sections(sections_path)
    parsed_sections_finetuned = compmusic_iamms.load_sections(sections_finetuned_path)

    # Check types
    assert type(parsed_sections) == annotations.SectionData
    assert type(parsed_sections.intervals) is np.ndarray
    assert type(parsed_sections.labels) is list
    assert type(parsed_sections_finetuned) == annotations.SectionData
    assert type(parsed_sections_finetuned.intervals) is np.ndarray
    assert type(parsed_sections_finetuned.labels) is list

    # Check values
    assert np.array_equal(
        parsed_sections.intervals[:, 0],
        np.array([0.835918, 89.083356, 96.636735]),
    )
    assert np.array_equal(
        parsed_sections.intervals[:, 1],
        np.array([2.565805, 90.661224, 98.086803]),
    )
    assert parsed_sections.labels == ["1000", "2000", "3000"]
    assert np.array_equal(
        parsed_sections_finetuned.intervals[:, 0],
        np.array([0.835918, 89.083356, 96.636735]),
    )
    assert np.array_equal(
        parsed_sections_finetuned.intervals[:, 1],
        np.array([2.565805, 90.661224, 98.086803]),
    )
    assert parsed_sections_finetuned.labels == ["1000", "2000", "3000"]

    assert compmusic_iamms.load_sections(None) is None


def test_load_nyas():
    data_home = "tests/resources/mir_datasets/compmusic_iamms"
    dataset = compmusic_iamms.Dataset(data_home, version="test")
    track = dataset.track("0_ALB_Bhairavi1_70albmgpnganesan")
    nyas_path = track.nyas_path
    parsed_nyas = compmusic_iamms.load_nyas(nyas_path)

    # Check types
    assert type(parsed_nyas) is annotations.EventData
    assert type(parsed_nyas.intervals) is np.ndarray
    assert type(parsed_nyas.events) is list

    # Check values
    assert np.array_equal(
        parsed_nyas.intervals,
        np.array(
            [
                [2.11882, 3.37560],
                [5.50023, 5.65116],
                [5.87465, 5.99075],
            ]
        ),
    )
    assert parsed_nyas.events == ["nyas", "nyas", "nyas"]
    assert compmusic_iamms.load_nyas(None) is None


def test_load_audio():
    data_home = "tests/resources/mir_datasets/compmusic_iamms"
    dataset = compmusic_iamms.Dataset(data_home, version="test")
    track = dataset.track("0_ALB_Bhairavi1_70albmgpnganesan")
    audio_path = track.audio_path
    audio, sr = compmusic_iamms.load_audio(audio_path)

    assert sr == 44100
    assert type(audio) == np.ndarray
    assert audio.shape[0] == 2

    assert compmusic_iamms.load_audio(None) is None
