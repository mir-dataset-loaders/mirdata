import csv
import json

from deprecated.sphinx import deprecated
import librosa
import numpy as np
import cv2

from mirdata import annotations, core, download_utils, io, jams_utils

class Track(core.Track):
    """Saraga Track Carnatic class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        audio_path (str): path to audio file
        audio_ghatam_path (str): path to ghatam audio file
        audio_mridangam_left_path (str): path to mridangam left audio file
        audio_mridangam_right_path (str): path to mridangam right audio file
        audio_violin_path (str): path to violin audio file
        audio_vocal_s_path (str): path to vocal s audio file
        audio_vocal_pat (str): path to vocal pat audio file
        ctonic_path (srt): path to ctonic annotation file
        pitch_path (srt): path to pitch annotation file
        pitch_vocal_path (srt): path to vocal pitch annotation file
        tempo_path (srt): path to tempo annotation file
        sama_path (srt): path to sama annotation file
        sections_path (srt): path to sections annotation file
        phrases_path (srt): path to phrases annotation file
        metadata_path (srt): path to metadata file

    Cached Properties:
        tonic (float): tonic annotation
        pitch (F0Data): pitch annotation
        pitch_vocal (F0Data): vocal pitch annotation
        tempo (dict): tempo annotations
        sama (BeatData): sama section annotations
        sections (SectionData): track section annotations
        phrases (SectionData): phrase annotations
        metadata (dict): track metadata with the following fields:

            - title (str): Title of the piece in the track
            - mbid (str): MusicBrainz ID of the track
            - album_artists (list, dicts): list of dicts containing the album artists present in the track and its mbid
            - artists (list, dicts): list of dicts containing information of the featuring artists in the track
            - raaga (list, dict): list of dicts containing information about the raagas present in the track
            - form (list, dict): list of dicts containing information about the forms present in the track
            - work (list, dicts): list of dicts containing the work present in the piece, and its mbid
            - taala (list, dicts): list of dicts containing the talas present in the track and its uuid
            - concert (list, dicts): list of dicts containing the concert where the track is present and its mbid

    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        # Audio path
        self.audio_path = self.get_path("audio-mix")
        self.video_path = self.get_path("video-mix")

        # Multitrack audio paths
        self.audio_ghatam_path = self.get_path("audio-ghatam")
        self.audio_mridangam_left_path = self.get_path("audio-mridangam-left")
        self.audio_mridangam_right_path = self.get_path("audio-mridangam-right")
        self.audio_violin_path = self.get_path("audio-violin")
        #self.audio_vocal_s_path = self.get_path("audio-vocal-s")
        self.audio_vocal_path = self.get_path("audio-vocal")

        # Gesture paths
        self.keypoints_path = self.get_path("keypoints")
        self.scores_path = self.get_path("scores")

        # Annotation paths
        #self.ctonic_path = self.get_path("ctonic")
        #self.pitch_path = self.get_path("pitch")
        #self.pitch_vocal_path = self.get_path("pitch-vocal")
        #self.tempo_path = self.get_path("tempo")
        #self.sama_path = self.get_path("sama")
        #self.sections_path = self.get_path("sections")
        #self.phrases_path = self.get_path("phrases")
        self.metadata_path = self.get_path("metadata")

    @core.cached_property
    def metadata(self):
        return load_metadata(self.metadata_path)


    @property
    def audio(self):
        """The track's audio

        Returns:
           * np.ndarray - audio signal
           * float - sample rate

        """
        return load_audio(self.audio_path)

    @property
    def video(self):
        """The track's audio

        Returns:
           * np.ndarray - audio signal
           * float - sample rate

        """
        return load_video(self.video_path)

    def to_jams(self):
        """Get the track's data in jams format

        Returns:
            jams.JAMS: the track's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            beat_data=[(self.sama, "sama")],
            f0_data=[(self.pitch, "pitch"), (self.pitch_vocal, "pitch_vocal")],
            section_data=[(self.sections, "sections")],
            event_data=[(self.phrases, "phrases")],
            metadata={
                "tempo": self.tempo,
                "tonic": self.tonic,
                "metadata": self.metadata,
            },
        )

def load_audio(audio_path):
    """Load a Saraga Carnatic audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    if audio_path is None:
        return None
    return librosa.load(audio_path, sr=44100, mono=False)

def load_video(video_path):
    """Load a Saraga Carnatic audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    if video_path is None:
        return None

    cap = cv2.VideoCapture('your_video.mp4')
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    video = np.array(frames)

    return video

def load_keypoints(keypoints_path):
    """Load a Saraga Carnatic audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    if keypoints_path is None:
        return None
    return np.load(keypoints_path)

def load_scores(scores_path):
    """Load a Saraga Carnatic audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    if scores_path is None:
        return None
    return np.load(scores_path)

@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The saraga_carnatic dataset
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
        reason="Use mirdata.datasets.saraga_carnatic.load_audio", version="0.3.4"
    )
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @deprecated(
        reason="Use mirdata.datasets.saraga_carnatic.load_audio", version="0.3.4"
    )
    def load_video(self, *args, **kwargs):
        return load_video(*args, **kwargs)

    @deprecated(
        reason="Use mirdata.datasets.saraga_carnatic.load_audio", version="0.3.4"
    )
    def load_keypoints(self, *args, **kwargs):
        return load_keypoints(*args, **kwargs)

    @deprecated(
        reason="Use mirdata.datasets.saraga_carnatic.load_audio", version="0.3.4"
    )
    def load_scores(self, *args, **kwargs):
        return load_scores(*args, **kwargs)

    @deprecated(
        reason="Use mirdata.datasets.saraga_carnatic.load_metadata", version="0.3.4"
    )
    def load_metadata(self, *args, **kwargs):
        return load_metadata(*args, **kwargs)
