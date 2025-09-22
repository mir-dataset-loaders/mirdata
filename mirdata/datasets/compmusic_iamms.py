"""Compmuic IAMMS Dataset Loader

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

import librosa
import numpy as np

from mirdata import annotations, core, download_utils, io

BIBTEX = """
@inproceedings{gulati2015improving,
  author    = {Sankalp Gulati and Joan Serr{\\`a} and Xavier Serra},
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
        filename="compmusic_iamms_index_1.0.json",
        url="https://zenodo.org/records/17175092/files/compmusic_iamms_index_1.0.json?download=1",
        checksum="3c8843f87b0fea83715058c5d8a84c22",
    ),
    "sample": core.Index(filename="compmusic_iamms_index_1.0_sample.json"),
}

REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="compmusic_iamms.zip",
        url="https://zenodo.org/records/16631794/files/MelodicSimilarityDataset.zip?download=1",
        checksum="d02c3f329558f91de2fe3bd613f6f2f5",
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
        label = "nyas"
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

    delimiter = "\t" if "\t" in first_line else " "
    reader = csv.reader(fhandle, delimiter=delimiter)

    for line in reader:
        times.append(float(line[0]))
        freqs.append(float(line[1]))

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
            name="compmusic_iamms",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    def load_sections(self, *args, **kwargs):
        return load_sections(*args, **kwargs)

    def load_nyas(self, *args, **kwargs):
        return load_sections(*args, **kwargs)

    def load_pitch(self, *args, **kwargs):
        return load_pitch(*args, **kwargs)

    def load_tonic(self, *args, **kwargs):
        return load_pitch(*args, **kwargs)
