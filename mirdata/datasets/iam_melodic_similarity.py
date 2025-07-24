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
        url="https://zenodo.org/records/16410081/files/iam_melodic_similarity_index.json?download=1",
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
    Track class for IAM Melodic Similarity dataset.

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None

    Attributes:
        audio_path (str): path to audio file
        sections_path (str): path to sections annotation file
        sections_finetuned_path (str): path to improved sections annotation file
        nyas_path (str): path to nyas features
        pitch_path (str): path to pitch annotation file
        pitch_finetuned_path (str): path to improved pitch annotation file
        tonic_path (str): path to tonic data file
        tonic_finetuned_path (str): path to improved tonic data file

    Cached Properties:
        audio (tuple): (audio signal as np.ndarray, sample rate as float)
        sections (SectionData): section annotations
        sections_finetuned (SectionData): improved section annotations
        nyas (EventData): nyas annotations
        pitch (F0Data): pitch annotations
        pitch_finetuned (F0Data): improved pitch annotations
        tonic (float): tonic
        tonic_finetuned (float): tonic finetuned
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.get_path("audio")
        self.sections_path = self.get_path("sections")
        self.sections_finetuned_path = self.get_path("sections-finetuned")
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
        return load_sections(self.sections_finetuned_path)

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
    """
    Load an audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        tuple: np.ndarray - the stereo audio signal, float - sample rate
    """
    if audio_path is None:
        return None
    return librosa.load(audio_path, sr=44100, mono=False)

@io.coerce_to_string_io
def load_nyas(fhandle):
    """
        Load a nyas annotation.

        Args:
            fhandle (str): path to annotation file

        Returns:
            EventData: nyas annotation intervals
    """
    intervals = []
    labels = []
    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        start = float(line[0])
        end = float(line[1])
        if start > end:
            continue
        label = 'nyas'
        intervals.append([start, end])
        labels.append(label)

    return annotations.EventData(np.array(intervals), "s", labels, "open")

@io.coerce_to_string_io
def load_sections(fhandle):
    """
        Load a sections annotation file.

        Args:
            fhandle (str): path to annotation file

        Returns:
            SectionData: section annotations with intervals (melodic phrasee) and labels (phrase identifier)
    """
    intervals = []
    labels = []
    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        start = float(line[0])
        end = float(line[1])
        if start > end:
            continue
        label = line[2]
        intervals.append([start, end])
        labels.append(label)

    return annotations.SectionData(np.array(intervals), "s", labels, "open")

@io.coerce_to_string_io
def load_pitch(fhandle):
    """
    Load pitch annotations.

    Args:
        fhandle (str): path to pitch file

    Returns:
        F0Data: pitch annotations
    """
    times = []
    freqs = []
    first_line = fhandle.readline()
    fhandle.seek(0)

    delimiter = '\t' if '\t' in first_line else ' '
    reader = csv.reader(fhandle, delimiter=delimiter)

    for line in reader:
        times.append(float(line[0]))
        freqs.append(float(line[1]))

    if not times:
        return None

    # Workaround for non-uniform time intervals in the dataset
    try:
        annotations.validate_uniform_times(times)
    except:
        time_diffs = np.diff(times)
        mean_time_diff = np.mean(time_diffs)
        times = np.arange(times[0], times[-1] + mean_time_diff, mean_time_diff)

    times = np.array(times)
    freqs = np.array(freqs)
    voicing = (freqs > 0).astype(float)
    return annotations.F0Data(times, "s", freqs, "hz", voicing, "binary")

@io.coerce_to_string_io
def load_tonic(fhandle):
    """
    Load track's tonic.

    Args:
        fhandle (str): path to tonic file

    Returns:
        float: tonic frequency in Hz
    """
    reader = csv.reader(fhandle, delimiter=" ")
    tonic = float(next(reader)[0])
    return tonic

@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The IAM Melodic Similarity dataset.

    This dataset contains Carnatic music recordings with annotations for
    sections, pitch, nyas, and tonic. It is designed to support research
    on melodic similarity with culturally relevant features.
    """
    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="iam_melodic_similarity",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @deprecated(
        reason="Use mirdata.datasets.iam_melodic_similarity.load_audio", version="1.0.0"
    )
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @deprecated(
        reason="Use mirdata.datasets.iam_melodic_similarity.load_sections", version="1.0.0"
    )
    def load_sections(self, *args, **kwargs):
        return load_sections(*args, **kwargs)

    @deprecated(
        reason="Use mirdata.datasets.iam_melodic_similarity.load_nyas", version="1.0.0"
    )
    def load_nyas(self, *args, **kwargs):
        return load_sections(*args, **kwargs)

    @deprecated(
        reason="Use mirdata.datasets.iam_melodic_similarity.load_pitch", version="1.0.0"
    )
    def load_pitch(self, *args, **kwargs):
        return load_pitch(*args, **kwargs)


    @deprecated(
        reason="Use mirdata.datasets.iam_melodic_similarity.load_tonic", version="1.0.0"
    )
    def load_tonic(self, *args, **kwargs):
        return load_pitch(*args, **kwargs)
