import os
import numpy as np
from mirdata import annotations
from mirdata.datasets import compmusic_iamms
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "0_Devi_Pavane"
    data_home = os.path.normpath("tests/resources/mir_datasets/saraga_audiovisual")
    dataset = compmusic_iamms.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "0_Devi_Pavane",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_audiovisual"),
            "saraga audio/Abhiram Bode/Devi Pavane",
            "Devi Pavane.wav",
        ),
        "video_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_audiovisual"),
            "saraga visual/Abhiram Bode/Devi Pavane",
            "Devi Pavane.mov",
        ),
        "audio_mridangam_left_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_audiovisual"),
            "saraga audio/Abhiram Bode/Devi Pavane",
            "Devi Pavane.multitrack-mridangam-left.wav",
        ),
        "audio_mridangam_right_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_audiovisual"),
            "saraga audio/Abhiram Bode/Devi Pavane",
            "Devi Pavane.multitrack-mridangam-right.wav",
        ),
        "audio_violin_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_audiovisual"),
            "saraga audio/Abhiram Bode/Devi Pavane",
            "Devi Pavane.multitrack-violin.wav",
        ),
        "audio_vocal_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_audiovisual"),
            "saraga audio/Abhiram Bode/Devi Pavane",
            "Devi Pavane.multitrack-vocal.wav",
        ),
        "keypoints_path": {
            "mridangam": os.path.join(
                os.path.normpath("tests/resources/mir_datasets/saraga_audiovisual"),
                "saraga gesture/Abhiram Bode/Devi Pavane/mridangam",
                "mridangam_0_2599_kpts.npy",
            ),
            "singer": os.path.join(
                os.path.normpath("tests/resources/mir_datasets/saraga_audiovisual"),
                "saraga gesture/Abhiram Bode/Devi Pavane/singer",
                "singer_0_2599_kpts.npy",
            ),
            "violin": os.path.join(
                os.path.normpath("tests/resources/mir_datasets/saraga_audiovisual"),
                "saraga gesture/Abhiram Bode/Devi Pavane/violin",
                "violin_0_2599_kpts.npy",
            ),
        }
        "scores_path": {
            "mridangam": os.path.join(
                os.path.normpath("tests/resources/mir_datasets/saraga_audiovisual"),
                "saraga gesture/Abhiram Bode/Devi Pavane/mridangam",
                "mridangam_0_2599_scores.npy",
            ),
            "singer": os.path.join(
                os.path.normpath("tests/resources/mir_datasets/saraga_audiovisual"),
                "saraga gesture/Abhiram Bode/Devi Pavane/singer",
                "singer_0_2599_scores.npy",
            ),
            "violin": os.path.join(
                os.path.normpath("tests/resources/mir_datasets/saraga_audiovisual"),
                "saraga gesture/Abhiram Bode/Devi Pavane/violin",
                "violin_0_2599_scores.npy",
            ),
        }
        "metadata_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_audiovisual"),
            "saraga metadata/Abhiram Bode/Devi Pavane",
            "Devi_Pavane.json",
        ),

    }

    expected_property_types = {
        "audio": np.ndarray,
        "video": np.ndarray,
        "audio_mridangam_left": np.ndarray,
        "audio_mridangam_right": np.ndarray,
        "audio_vocal": np.ndarray,
        "audio_violin": np.ndarray,
        "mridangam_gesture": annotations.GestureData,
        "singer_gesture": annotations.GestureData,
        "violin_gesture": annotations.GestureData,
        "metadata": dict,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 44100
    assert audio.shape[0] == 2


def test_load_audio():
    data_home = "tests/resources/mir_datasets/saraga_audiovisual"
    dataset = saraga_audiovisual.Dataset(data_home, version="test")
    track = dataset.track("0_Devi_Pavane")
    audio_path = track.audio_path
    audio, sr = saraga_audiovisual.load_audio(audio_path)

    assert sr == 44100
    assert type(audio) == np.ndarray
    assert audio.shape[0] == 2

    assert saraga_audiovisual.load_audio(None) is None

def test_load_video():
    data_home = "tests/resources/mir_datasets/saraga_audiovisual"
    dataset = saraga_audiovisual.Dataset(data_home, version="test")
    track = dataset.track("0_Devi_Pavane")
    video_path = track.video_path
    video, fps = saraga_audiovisual.load_video(video_path)

    assert type(video) == np.ndarray
    assert type(fps) == float

    assert saraga_audiovisual.load_video(None) is None

def test_load_metadtata(metadata_path):
    data_home = "tests/resources/mir_datasets/saraga_audiovisual"
    dataset = saraga_audiovisual.Dataset(data_home, version="test")
    track = dataset.track("0_Devi_Pavane")
    metadata_path = track.metadata_path
    parsed_metadata = saraga_audiovisual.load_metadata(metadata_path)


    assert parsed_metadata["mbid"] == "bf197cad-30d5-478c-9eae-b0897eef7a4f"
    assert parsed_metadata["title"] == "Devi Pavane"
    assert parsed_metadata["length"] == 2599.0333333333333
    assert parsed_metadata["artists"] == [
        {
            "mbid": "bf197cad-30d5-478c-9eae-b0897eef7a4f",
            "name": "Abhiram Bode",
            "instrument": {
                "mbid": "d92884b7-ee0c-46d5-96f3-918196ba8c5b",
                "name": "Voice"
            },
            "lead": True,
            "attributes": "lead vocals"
        },
        {
            "mbid": "9b8912a2-afa1-4a14-8c5f-5225421a2bb5",
            "name": "Sivateja",
            "instrument": {
                "mbid": "089f123c-0f7d-4105-a64e-49de81ca8fa4",
                "name": "Violin"
            },
            "lead": False,
            "attributes": ""
        },
        {
            "mbid": "b5b3f876-cd38-4d50-b1ad-ac9e18ad73d4",
            "name": "Adudurai Guruprasad",
            "instrument": {
                "mbid": "f689271c-37bc-4c49-92a3-a14b15ee5d0e",
                "name": "Mridangam"
            },
            "lead": False,
            "attributes": ""
        }
    ]
    assert parsed_metadata["raaga"] == [
        {
            "uuid": "",
            "name": "Saveri"
        }
    ]
    assert parsed_metadata["taala"] == [
        {
            "uuid": "",
            "name": "Adi"
        }
    ]
    assert parsed_metadata["form"] == []
    assert parsed_metadata["work"] == []
    assert parsed_metadata["concert"] == []
    assert parsed_metadata["album_artists"] == []

def test_load_gesture():
    data_home = "tests/resources/mir_datasets/saraga_audiovisual"
    dataset = saraga_audiovisual.Dataset(data_home, version="test")
    track = dataset.track("0_Devi_Pavane")
    track = dataset.track("0_Devi_Pavane")
    keypoints_path = track.keypoints_path['singer']
    scores_path = scores.keypoints_path['singer']
    gesture = saraga_audiovisual.load_gesture(keypoints_path, scores_path)

    assert gesture.keypoints == np.array([
        [100, 200],
        [200, 400]
    ])

    assert gesture.score == np.array([
        [1, 0.5]
    ])
