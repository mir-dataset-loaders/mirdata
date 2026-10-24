"""
Saraga Audiovisual Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    Saraga Audiovisual includes diverse renditions of Carnatic vocal performances, totalling 42 concerts and more than 60 hours of music. It includes video recordings for all concerts, allowing for a wide range of multimodal analyses.
    It also contains high-quality human pose estimation data of the musicians extracted from the video footage, and perform benchmarking experiments for the different modalities to validate the utility of the novel collection.

    The dataset contains a total of 233 tracks.

    - Audio recordings (multitrack and mix): Vocal, violin, mridangam (left and right), and mix.
    - Video recordings of the performances.
    - Human pose estimation data for the musicians, including keypoints and confidence scores.
    - Metadata for each track, including information about the artist, composition, and performance context.

    The files of this dataset are shared with the following license:
    Creative Commons Attribution Non Commercial Share Alike 4.0 International

    Dataset compiled by: Sivasankar, A.

    For more information about the dataset as well as Compmusic and annotations, please refer to:
    https://zenodo.org/records/17405610, where a really detailed explanation of the dataset is published.
"""

import json
import logging
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np

from mirdata import core, download_utils, io

try:
    from moviepy import VideoFileClip
except ImportError:
    logging.error(
        "In order to use saraga_audiovisual you must have MoviePy installed. "
        "Please reinstall mirdata using `pip install 'mirdata[saraga_audiovisual]'"
    )
    raise

BIBTEX = """
@dataset{sivasankar2024saraga,
  author       = {A. S. Sivasankar},
  title        = {Saraga Audiovisual: a large multimodal open data collection for the analysis of Carnatic music},
  year         = {2024},
  month        = {November},
  day          = {10},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17405610},
  url          = {https://doi.org/10.5281/zenodo.17405610}
}
"""

INDEXES = {
    "default": "1.0",
    "test": "sample",
    "1.0": core.Index(
        filename="saraga_audiovisual_index.json",
        url="https://zenodo.org/records/18291024/files/saraga_audiovisual_index.json?download=1",
        checksum="b847ca946f2a88956569c897b186a148",
    ),
    "sample": core.Index(filename="saraga_audiovisual_index_1.0_sample.json"),
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
        url="https://zenodo.org/records/17405610/files/saraga%20visual.zip?download=1",
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
    """Saraga Audiovisual Track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        audio_path (str): path to the mix audio file
        audio_mridangam_left_path (str): path to mridangam left audio file
        audio_mridangam_right_path (str): path to mridangam right audio file
        audio_violin_path (str): path to violin audio file
        audio_vocal_path (str): path to vocal audio file
        video_path (str): path to video file
        keypoint_paths (dict): paths to keypoint files, keyed by "mridangam", "singer" and "violin"
        score_paths (dict): paths to confidence score files, keyed by "mridangam", "singer" and "violin"
        metadata_path (str): path to metadata file

    Cached Properties:
        metadata (dict): track metadata
        mridangam_gesture (tuple): keypoints and scores for the mridangam player
        singer_gesture (tuple): keypoints and scores for the singer
        violin_gesture (tuple): keypoints and scores for the violinist

    Properties:
        audio (tuple): mix audio signal and sample rate
        audio_mridangam_left (tuple): mridangam left audio signal and sample rate
        audio_mridangam_right (tuple): mridangam right audio signal and sample rate
        audio_violin (tuple): violin audio signal and sample rate
        audio_vocal (tuple): vocal audio signal and sample rate
        video (tuple): video frames and frame rate

    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(
            track_id,
            data_home,
            dataset_name=dataset_name,
            index=index,
            metadata=metadata,
        )

        self.audio_path = self.get_path("audio-mix")
        self.audio_mridangam_left_path = self.get_path("audio-mridangam-left")
        self.audio_mridangam_right_path = self.get_path("audio-mridangam-right")
        self.audio_violin_path = self.get_path("audio-violin")
        self.audio_vocal_path = self.get_path("audio-vocal")
        self.video_path = self.get_path("video")

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

        self.metadata_path = self.get_path("metadata")

    @core.cached_property
    def metadata(self) -> Optional[dict]:
        return load_metadata(self.metadata_path)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The mix audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    @property
    def audio_mridangam_left(self) -> Optional[Tuple[np.ndarray, float]]:
        """The mridangam left audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_mridangam_left_path)

    @property
    def audio_mridangam_right(self) -> Optional[Tuple[np.ndarray, float]]:
        """The mridangam right audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_mridangam_right_path)

    @property
    def audio_violin(self) -> Optional[Tuple[np.ndarray, float]]:
        """The violin audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_violin_path)

    @property
    def audio_vocal(self) -> Optional[Tuple[np.ndarray, float]]:
        """The vocal audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_vocal_path)

    @property
    def video(self) -> Optional[Tuple[np.ndarray, float]]:
        """The video

        Returns:
            * np.ndarray - video frames (frames, height, width, channels)
            * float - frame rate

        """
        return load_video(self.video_path)

    @core.cached_property
    def mridangam_gesture(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return load_gesture(
            self.keypoint_paths["mridangam"], self.score_paths["mridangam"]
        )

    @core.cached_property
    def singer_gesture(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return load_gesture(self.keypoint_paths["singer"], self.score_paths["singer"])

    @core.cached_property
    def violin_gesture(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return load_gesture(self.keypoint_paths["violin"], self.score_paths["violin"])


@io.coerce_to_string_io
def load_metadata(fhandle: TextIO) -> dict:
    """Load a Saraga Audiovisual metadata file

    Args:
        fhandle (str or file-like): File-like object or path to metadata json

    Returns:
        dict: metadata of the track

    """
    return json.load(fhandle)


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a Saraga Audiovisual audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the audio signal
        * float - The sample rate of the audio file

    """
    return librosa.load(fhandle, sr=None, mono=False)


def load_video(video_path: str) -> Optional[Tuple[np.ndarray, float]]:
    """Load a Saraga Audiovisual video file.

    Args:
        video_path (str): path to video file

    Returns:
        * np.ndarray - the video frames (frames, height, width, channels)
        * float - The frame rate of the video file

    """
    if video_path is None:
        return None

    with VideoFileClip(video_path) as clip:
        frames = np.array(list(clip.iter_frames()))
        return frames, clip.fps


def load_gesture(
    keypoints_path: str, scores_path: str
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load a Saraga Audiovisual gesture file.

    Args:
        keypoints_path (str): path to keypoints file
        scores_path (str): path to scores file

    Returns:
        * np.ndarray - the keypoints of the pose estimation
        * np.ndarray - the confidence scores of the keypoints

    """
    if keypoints_path is None or scores_path is None:
        return None

    return np.load(keypoints_path), np.load(scores_path)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The Saraga Audiovisual dataset
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
