"""Saraga Audiovisual Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    This dataset comprises audio excerpts and manually done annotations of the melodic phrases in Carnatic and Hindustani music.
    This dataset can be used to develop and evaluate approaches for computing melodic similarity between short-time melodic patterns in Indian art music.

    The dataset contains the following manual annotations referring to audio files:

    - Section annotations, both original and finetuned, stored as start and end timestamps together with the phrase ID of the section (similar melodic phrases have the same ID).
    - Nyas event annotations stored as start and end timestamps.
    - Audio features automatically extracted and stored: pitch and tonic.
    - The annotations are stored in files with song identifier as the filename and file extension:
        - Section annotations: `.anot` and `.anotEdit`
        - Nyas annotations: `.flatSegNyas`
        - Pitch annotations: `.pitch`, `.pitchSilIntrpPP`, `tpe` and `tpe5msSilIntrpPP`
        - Tonic: `.tonic` and `.tonic`

    The dataset contains a total of 32 tracks.

    The files of this dataset are shared with the following license:
    Creative Commons Attribution Non Commercial Share Alike 4.0 International

    Dataset compiled by: Gulati, S., Serrà, J., and Serra, X.

    For more information about the dataset as well as IAM and annotations, please refer to:
    https://zenodo.org/records/16631794, where a really detailed explanation of the data and annotations is published.

"""

import csv
import json
import os

import cv2
import librosa
import numpy as np
from deprecated.sphinx import deprecated

from mirdata import annotations, core, download_utils, io

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
        url="https://drive.google.com/uc?export=download&id=1QYifrzCzPSJTh813HJaTph4eLx8TzBQY",  # TODO
        checksum="4f3a8e919593aa2b71f7a0b81cc8cc00",  # TODO
    ),
    "sample": core.Index(filename="saraga_audiovisual_index_sample.json"),
}

REMOTES = {
    "metadata": download_utils.RemoteFileMetadata(
        filename="saraga metadata.zip",
        url="https://zenodo.org/records/17405610/files/saraga%20metadata.zip?download=1",
        checksum="1f5cd4b1287d07a87e8dd51a178dd0a1",
    ),
    "audio": download_utils.RemoteFileMetadata(
        filename="saraga audio.zip",
        url="https://zenodo.org/records/17405610/files/saraga%20audio.zip?download=1",
        checksum="ba93a85d9dc6e844177ea4a6c830eeeb",
    ),
    "visual": download_utils.RemoteFileMetadata(
        filename="saraga visual.zip",
        url="https://zenodo.org/records/15102483/files/saraga%20visual.zip?download=1",
        checksum="067b635d1fedb82e8261dcc1237a469f",
    ),
    "gesture": download_utils.RemoteFileMetadata(
        filename="saraga gesture.zip",
        url="https://zenodo.org/records/17405610/files/saraga%20gesture.zip?download=1",
        checksum="6f2700caf088293ea50ba455b3407f10",
    ),
}

LICENSE_INFO = (
    "Creative Commons Attribution Non Commercial Share Alike 4.0 International."
)


class Track(core.Track):
    """
    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        audio_path (str): path to audio file
        audio_mridangam_left_path (str): path to mridangam left audio file
        audio_mridangam_right_path (str): path to mridangam right audio file
        audio_violin_path (str): path to violin audio file
        audio_vocal_path (str): path to vocal audio file
        video_path (srt): path to video file
        keypoints_path (dict): paths to keypoint annotation files
        scores_path (dict): paths to scores annotation files
        metadata_path (srt): path to metadata file

    Cached Properties:
        audio (numpy.ndarray, float): audio, samplerate
        audio_mridangam_left (numpy.ndarray, float): mridangam left audio, samplerate
        audio_mridangam_right (numpy.ndarray, float): mridangam right audio, samplerate
        audio_violin (numpy.ndarray, float): violin audio, samplerate
        audio_vocal (numpy.ndarray, float): vocal audio, samplerate
        pitch (numpy.ndarray, float): video, framerate
        mridangam_gesture (GesturData): gesture annotation for mridangam
        singer_gesture (GesturData): gesture annotation for singer
        violin_gesture (GesturData): gesture annotation for violin
        metadata (dict): track metadata
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
            "mridangam": self.get_path("keypoints-mridangam"),
            "singer": self.get_path("keypoints-singer"),
            "violin": self.get_path("keypoints-violin"),
        }
        self.score_paths = {
            "mridangam": self.get_path("scores-mridangam"),
            "singer": self.get_path("scores-singer"),
            "violin": self.get_path("scores-violin"),
        }

        # Metadata path
        self.metadata_path = self.get_path("metadata")

    @core.cached_property
    def metadata(self):
        return load_metadata(self.metadata_path)

    @core.cached_property
    def audio(self):
        return load_audio(self.audio_path)

    @core.cached_property
    def audio_mridangam_left(self):
        return load_audio(self.audio_mridangam_left_path)

    @core.cached_property
    def audio_mridangam_right(self):
        return load_audio(self.audio_mridangam_right_path)

    @core.cached_property
    def audio_vocal(self):
        return load_audio(self.audio_vocal_path)

    @core.cached_property
    def audio_violin(self):
        return load_audio(self.audio_violin_path)

    @core.cached_property
    def video(self):
        return load_video(self.video_path)

    @core.cached_property
    def mridangam_gesture(self):
        return load_gesture(
            self.keypoint_paths["mridangam"], self.score_paths["mridangam"]
        )

    @core.cached_property
    def singer_gesture(self):
        return load_gesture(self.keypoint_paths["singer"], self.score_paths["singer"])

    @core.cached_property
    def violin_gesture(self):
        return load_gesture(self.keypoint_paths["violin"], self.score_paths["violin"])


@io.coerce_to_string_io
def load_metadata(fhandle):
    """Load a Saraga Audiovisual metadata file

    Args:
        fhandle (str or file-like): File-like object or path to metadata json

    Returns:
        dict: metadata with the following fields
    """
    if not fhandle:
        return None

    return json.load(fhandle)


def load_audio(audio_path):
    """Load a Saraga Audiovisual audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    if audio_path is None:
        return None

    if not os.path.exists(audio_path):
        return None

    return librosa.load(audio_path, sr=44100, mono=False)


def load_video(video_path):
    """Load a Saraga Audiovisual video file.

    Args:
        video_path (str): path to video file

    Returns:
        * np.ndarray - the video signal (frames, height, width, channels)
        * float - The frame rate of the video file

    """
    if video_path is None:
        return None

    if not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
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


def load_gesture(keypoints_path, scores_path):
    """Load a Saraga Audiovisual gesture file.

    Args:
        keypoints_path (str): path to keypoints file
        scores_path (str): path to scores file

    Returns:
        GestureData - gesture data

    """
    if keypoints_path is None or scores_path is None:
        return None

    if not os.path.exists(keypoints_path) or not os.path.exists(scores_path):
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

    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    def load_video(self, *args, **kwargs):
        return load_video(*args, **kwargs)

    def load_mridangam_gesture(self, *args, **kwargs):
        return load_gesture(*args, **kwargs)

    def load_singer_gesture(self, *args, **kwargs):
        return load_gesture(*args, **kwargs)

    def load_violin_gesture(self, *args, **kwargs):
        return load_gesture(*args, **kwargs)

    def load_metadata(self, *args, **kwargs):
        return load_metadata(*args, **kwargs)
