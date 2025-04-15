import csv
import json

from deprecated.sphinx import deprecated
import librosa
import numpy as np
import cv2

from mirdata import annotations, core, download_utils, io, jams_utils

BIBTEX = """
@dataset{sivasankar2024saraga,
  author       = {A. S. Sivasankar},
  title        = {Saraga Audiovisual: a large multimodal open data collection for the analysis of Carnatic music},
  year         = {2024},
  month        = {November},
  day          = {10},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15102483},
  url          = {https://doi.org/10.5281/zenodo.15102483}
}
"""

INDEXES = {
    "default": "1.0",
    "test": "sample",
    "1.0": core.Index(
        filename="saraga_audiovisual_index.json",
        url="https://drive.google.com/uc?export=download&id=1QYifrzCzPSJTh813HJaTph4eLx8TzBQY",
        checksum="4cac461c0baba0dde95061d5bc84a875",
    ),
    "sample": core.Index(filename="saraga_carnatic.json"),
}

REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="saraga_audiovisual.zip",
        url="https://drive.google.com/uc?export=download&id=1CDXIqqjAvB28Z8vH-AzAiJ6Rw-VvjV7j",
        checksum="4cac461c0baba0dde95061d5bc84a875",
    )
}

LICENSE_INFO = (
    "Creative Commons Attribution Non Commercial Share Alike 4.0 International."
)

class Track(core.Track):
    """
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        # Audio path
        self.audio_path = self.get_path("audio-mix")
        self.video_path = self.get_path("video")

        # Multitrack audio paths
        self.audio_mridangam_left_path = self.get_path("audio-mridangam-left")
        self.audio_mridangam_right_path = self.get_path("audio-mridangam-right")
        self.audio_violin_path = self.get_path("audio-violin")
        self.audio_vocal_path = self.get_path("audio-vocal")

        # Gesture paths
        self.keypoint_paths = {
            'mridangam': self.get_path("keypoints-mridangam"),
            'singer': self.get_path("keypoints-singer"),
            'violin': self.get_path("keypoints-violin"),
        }
        self.score_paths = {
            'mridangam': self.get_path("scores-mridangam"),
            'singer': self.get_path("scores-singer"),
            'violin': self.get_path("scores-violin"),
        }

        # Metadata path
        self.metadata_path = self.get_path("metadata")

    @core.cached_property
    def metadata(self):
        return load_metadata(self.metadata_path)

    @property
    def audio(self):
        return load_audio(self.audio_path)

    @property
    def video(self):
        return load_video(self.video_path)

    @property
    def mridangam_gesture(self):
        return load_mridangam_gesture(self.keypoint_paths['mridangam'], self.score_paths['mridangam'])

    @property
    def singer_gesture(self):
        return load_singer_gesture(self.keypoint_paths['singer'], self.score_paths['singer'])

    @property
    def violin_gesture(self):
        return load_violin_gesture(self.keypoint_paths['violin'], self.score_paths['violin'])


@io.coerce_to_string_io
def load_metadata(fhandle):
    """
    """
    return json.load(fhandle)

def load_audio(audio_path):
    """
    """
    if audio_path is None:
        return None
    return librosa.load(audio_path, sr=44100, mono=False)

def load_video(video_path):
    """
    """
    if video_path is None:
        return None

    cap = cv2.VideoCapture('your_video.mp4')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    video = np.array(frames)

    return video, fps

def load_mridangam_gesture(keypoints_path, scores_path):
    """
    """
    if keypoints_path is None or scores_path is None:
        return None

    keypoints = np.load(keypoints_path)
    scores = np.load(scores_path)

    gesture = annotations.GestureData(keypoints, scores)

    return gesture

def load_singer_gesture(keypoints_path, scores_path):
    """
    """
    if keypoints_path is None or scores_path is None:
        return None

    keypoints = np.load(keypoints_path)
    scores = np.load(scores_path)

    gesture = annotations.GestureData(keypoints, scores)

    return gesture

def load_violin_gesture(keypoints_path, scores_path):
    """
    """
    if keypoints_path is None or scores_path is None:
        return None

    keypoints = np.load(keypoints_path)
    scores = np.load(scores_path)

    gesture = annotations.GestureData(keypoints, scores)

    return gesture


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The saraga_audiovisual dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="saraga_audiovisual",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @deprecated(
        reason="Use mirdata.datasets.saraga_audiovisual.load_audio", version="0.3.4"
    )
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @deprecated(
        reason="Use mirdata.datasets.saraga_audiovisual.load_video", version="0.3.4"
    )
    def load_video(self, *args, **kwargs):
        return load_video(*args, **kwargs)

    @deprecated(
        reason="Use mirdata.datasets.saraga_audiovisual.load_mridangam_gesture", version="0.3.4"
    )
    def load_mridangam_gesture(self, *args, **kwargs):
        return load_mridangam_gesture(*args, **kwargs)

    @deprecated(
        reason="Use mirdata.datasets.saraga_audiovisual.load_singer_gesture", version="0.3.4"
    )
    def load_singer_gesture(self, *args, **kwargs):
        return load_singer_gesture(*args, **kwargs)

    @deprecated(
        reason="Use mirdata.datasets.saraga_audiovisual.load_violin_gesture", version="0.3.4"
    )
    def load_violin_gesture(self, *args, **kwargs):
        return load_violin_gesture(*args, **kwargs)

    @deprecated(
        reason="Use mirdata.datasets.saraga_audiovisual.load_metadata", version="0.3.4"
    )
    def load_metadata(self, *args, **kwargs):
        return load_metadata(*args, **kwargs)
