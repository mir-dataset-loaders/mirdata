import csv
import json

from deprecated.sphinx import deprecated
import librosa
import numpy as np

from mirdata import annotations, core, download_utils, io, jams_utils

BIBTEX = """
@inproceedings{gulati2015improving,
  author    = {Sankalp Gulati and Joan Serr{\`a} and Xavier Serra},
  title     = {Improving melodic similarity in Indian art music using culture-specific melodic characteristics},
  booktitle = {Proceedings of the 16th International Society for Music Information Retrieval Conference (ISMIR)},
  pages     = {680--686},
  year      = {2015},
  address   = {Malaga, Spain}
}

"""

INDEXES = {
    "default": "1.0",
    "test": "sample",
    "1.0": core.Index(
        filename="iam_melodic_similarity_index.json",
        url="https://zenodo.org/records/15351055/files/iam_melodic_similarity_index.json?download=1",
        checksum="7f968243e8acebaa6cbba05cf9218ae6",
    ),
    "sample": core.Index(filename="iam_melodic_similarity_index_sample.json"),
}

REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="iam_melodic_similarity.zip",
        url="https://zenodo.org/records/15350958/files/MelodicSimilarityDataset.zip?download=1",
        checksum="96d35e07be5e8d5b680efac5248d37d2",
    )
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
        pitch (numpy.ndarray, float): video, framerate
        mridangam_gesture (GesturData): gesture annotation for mridangam
        singer_gesture (GesturData): gesture annotation for singer
        violin_gesture (GesturData): gesture annotation for violin
        metadata (dict): track metadata
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        # Audio path
        self.audio_path = self.get_path("audio")

        # Annotation paths
        self.sections_path = self.get_path("sections")
        self.sections_finetuned_path = self.get_path("sections-finetuned")

        # Feature paths
        self.nyas_path = self.get_path("nyas")
        self.pitch_path = self.get_path("pitch")
        self.pitch_finetuned_path = self.get_path("pitch-finetuned")
        self.tonic_path = self.get_path("tonic")
        self.tonic_finetuned_path = self.get_path("tonic-finetuned")

    @property
    def audio(self):
        return load_audio(self.audio_path)

    @core.cached_property
    def sections(self):
        return load_sections(self.sections_path)

    @core.cached_property
    def sections_finetuned(self):
        return load_pitch(self.sections_finetuned_path)

    @core.cached_property
    def nyas(self):
        return load_nyas(self.nyas_path)

    @core.cached_property
    def pitch(self):
        return load_pitch(self.pitch_path)

    @core.cached_property
    def pitch_finetuned(self):
        return load_pitch(self.pitch_finetuned_path)

    @core.cached_property
    def tonic(self):
        return load_tonic(self.tonic_path)

    @core.cached_property
    def tonic_finetuned(self):
        return load_tonic(self.tonic_finetuned_path)


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
    return librosa.load(audio_path, sr=44100, mono=False)

@io.coerce_to_string_io
def load_nyas(fhandle):
    """Load a Saraga Audiovisual mridangam gesture file.

    Args:
        fhandle (str): path to annotation file

    Returns:
        * SectionData - annotations

    """
    intervals = []
    section_labels = []
    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        start = float(line[0])
        end = float(line[1])
        label = 'Nyas'
        intervals.append([start, end])
        section_labels.append(label)

    return annotations.SectionData(np.array(intervals), "s", section_labels, "open")

@io.coerce_to_string_io
def load_sections(fhandle):
    """Load a Saraga Audiovisual mridangam gesture file.

    Args:
        fhandle (str): path to annotation file

    Returns:
        * SectionData - annotations

    """
    intervals = []
    section_labels = []
    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        start = float(line[0])
        end = float(line[1])
        label = line[2]
        intervals.append([start, end])
        section_labels.append(label)

    return annotations.SectionData(np.array(intervals), "s", section_labels, "open")

@io.coerce_to_string_io
def load_pitch(fhandle):
    """Load pitch

    Args:
        fhandle (str or file-like): Local path where the pitch annotation is stored.

    Returns:
        F0Data: pitch annotation

    """
    times = []
    freqs = []

    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        times.append(float(line[0]))
        freqs.append(float(line[1]))

    if not times:
        return None

    times = np.array(times)
    freqs = np.array(freqs)
    voicing = (freqs > 0).astype(float)
    return annotations.F0Data(times, "s", freqs, "hz", voicing, "binary")

@io.coerce_to_string_io
def load_tonic(fhandle):
    """Load track absolute tonic

    Args:
        fhandle (str or file-like): Local path where the tonic path is stored.

    Returns:
        int: Tonic annotation in Hz

    """
    reader = csv.reader(fhandle, delimiter=" ")
    tonic = float(next(reader)[0])
    return tonic

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
        reason="Use mirdata.datasets.saraga_audiovisual.load_sections", version="0.3.4"
    )
    def load_sections(self, *args, **kwargs):
        return load_sections(*args, **kwargs)


    @deprecated(
        reason="Use mirdata.datasets.saraga_audiovisual.load_pitch", version="0.3.4"
    )
    def load_pitch(self, *args, **kwargs):
        return load_pitch(*args, **kwargs)


    @deprecated(
        reason="Use mirdata.datasets.saraga_audiovisual.load_tonic", version="0.3.4"
    )
    def load_tonic(self, *args, **kwargs):
        return load_pitch(*args, **kwargs)
