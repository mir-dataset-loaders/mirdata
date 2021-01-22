# -*- coding: utf-8 -*-
"""CompMusic Carnatic Varnam Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    Carnatic varnam dataset is a collection of 28 solo vocal recordings, recorded for our research on intonation
    analysis of Carnatic raagas. The collection has the audio recordings, taala cycle annotations and notations in a
    machine readable format.

    **Audio music content**
    They feature 7 varnams in 7 rāgas sung by 5 young professional singers who received training for more than 15 years.
    They are all set to Adi taala. Measuring the intonation variations require absolutely clean pitch contours. For
    this, all the varṇaṁs are recorded without accompanying instruments, except the drone.

    **Taala annotations**
    The recordings are annotated with taala cycles, each annotation marking the starting of a cycle. We have later
    automatically divided each cycle into 8 equal parts. The annotations are made available as sonic visualizer
    annotation layers. Each annotation is of the format m.n where m is the cycle number and n is the division within
    the cycle. All m.1 annotations are manually done, whereas m.[2-8] are automatically labelled.

    **Notations**
    The notations for 7 varnams are procured from an archive curated by Shivkumar, in word document format. They are
    manually converted to a machine readable format (yaml). Each file is essentially a dictionary with section names
    of the composition as keys. Each section is represented as a list of cycles. Each cycle in turn has a list of
    divisions.

    **Sections**
    From the information inferred from both Taala and Notations, we have included Section annotations in this loader.
    These sections refer to the typical Carnatic Varnam structure.

    **Possible uses of the dataset**
    The distinct advantage of this dataset is the free availability of the audio content. Along with the annotations,
    it can be used for melodic analyses: characterizing intonation, motif discovery and tonic identification. The
    availability of a machine readable notation files allows the dataset to be used for audio-score alignment.
"""

import numpy as np
import os
from xml.dom import minidom
import logging
import librosa
import csv

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core
from mirdata import annotations

BIBTEX = """
@dataset{koduri_g_k_2014_1257118,
  author       = {Koduri, G. K. and
                  Ishwar, V. and
                  Serrà, J. and
                  Serra, X.},
  title        = {Carnatic Varnam Dataset},
  month        = feb,
  year         = 2014,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.1257118},
  url          = {https://doi.org/10.5281/zenodo.1257118}
}
"""

REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="saraga1.5_carnatic.zip",
        url="https://zenodo.org/record/4301737/files/saraga1.5_carnatic.zip?download=1",
        checksum="e4fcd380b4f6d025964cd16aee00273d",
        destination_dir=None,
    )
}

LICENSE_INFO = (
    "Creative Commons Attribution Non Commercial No Derivatives 4.0 International"
)


def _load_metadata(metadata_path):

    if not os.path.exists(metadata_path):
        logging.info("Metadata file {} not found.".format(metadata_path))
        return None

    metadata = {}
    with open(metadata_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=":")
        for line in reader:
            metadata[line[0]] = float(line[1].split(' ')[1])

    metadata['data_home'] = metadata_path.split('compmusic_carnatic_varnam')[0]

    return metadata


DATA = core.LargeData("compmusic_carnatic_varnam_index.json", _load_metadata)


class Track(core.Track):
    """CompMusic Carnatic Varnam Track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        tonic (float): float identifying the absolute tonic of the track
        artist (str): string identifying the performing artist in the track
        raaga (str): string identifying the raaga present in the track

    Cached Properties:
        taala (BeatData): taala annotations
        notation (EventData): note notations in IAM solfège symbols representation
        sections (SectionData): track section annotations

    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index["tracks"]:
            raise ValueError(
                "{} is not a valid track ID in Saraga Carnatic".format(track_id)
            )

        self.track_id = track_id

        self._data_home = data_home
        self._track_paths = DATA.index["tracks"][track_id]

        # Audio path
        self.audio_path = os.path.join(
            self._data_home, self._track_paths["audio"][0]
        )

        # Annotation paths
        self.taala_path = core.none_path_join(
            [self._data_home, self._track_paths["taala"][0]]
        )
        self.notation_path = core.none_path_join(
            [self._data_home, self._track_paths["notation"][0]]
        )
        self.metadata_path = core.none_path_join(
            [self._data_home, DATA.index["metadata"][0]]
        )

        # -- Track attributes --
        # Load metadata (containing tonic information)
        metadata = DATA.metadata(self.metadata_path)
        if metadata is not None:
            self.tonic = metadata[self.track_id.split('_')[0]]
        else:
            self.tonic = None
        self.artist = self.track_id.split('_')[0]
        self.raaga = self.track_id.split('_')[1]

    @core.cached_property
    def taala(self):
        return load_taala(self.taala_path)

    @core.cached_property
    def notation(self):
        return load_notation(self.notation_path, self.taala_path)

    @core.cached_property
    def sections(self):
        return load_sections(self.notation_path, self.taala_path)

    @property
    def audio(self):
        """The track's audio

        Returns:
           * np.ndarray - audio signal
           * float - sample rate

        """
        return load_audio(self.audio_path)

    def to_jams(self):
        """Get the track's data in jams format

        Returns:
            jams.JAMS: the track's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            beat_data=[(self.taala, "taala")],
            section_data=[(self.sections, "sections")],
            event_data=[(self.notation, "notation")],
            metadata={
                "performer": self.artist,
                "raaga": self.raaga,
                "tonic": self.tonic
            },
        )


def load_audio(audio_path):
    """Load a CompMusic Carnatic Varnam audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    if audio_path is None:
        return None

    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=22100, mono=False)


def load_taala(taala_path):
    """Load taala annotation

    Args:
        taala_path (str): Local path where the taala annotation is stored.

    Returns:
        BeatData: taala annotation for track
    """
    if taala_path is None:
        return None

    if not os.path.exists(taala_path):
        raise IOError("taala_path {} does not exist".format(taala_path))

    # Load svl file
    dom = minidom.parse(taala_path)

    # Load data
    data = dom.getElementsByTagName('data')[0]

    # Store points and calculate total length
    points = data.getElementsByTagName('dataset')[0].getElementsByTagName('point')
    num_points = len(points)

    # Parse sampling frequency
    fs = float(data.getElementsByTagName('model')[0].getAttribute('sampleRate'))

    beat_times = []
    beat_positions = []
    for beat in range(num_points):
        beat_times.append(float(points[beat].getAttribute('frame')) / fs)
        beat_positions.append(1)

    return annotations.BeatData(np.array(beat_times), np.array(beat_positions))


def load_notation(notation_path, taala_path):
    """Load notation (notes)

    Args:
        notation_path (str): Local path where the phrase annotation is stored.
            If `None`, returns None.
        taala_path (str): Local path where the taala annotation is stored.

    Returns:
        EventData: melodic notation for track

    """
    if notation_path is None:
        return None
    if taala_path is None:
        return None

    if not os.path.exists(notation_path):
        raise IOError("notation_path {} does not exist".format(notation_path))
    if not os.path.exists(taala_path):
        raise IOError("taala_path {} does not exist".format(taala_path))

    start_times = []
    end_times = []
    events = []

    dom = minidom.parse(taala_path)
    data = dom.getElementsByTagName('data')[0]
    points = data.getElementsByTagName('dataset')[0].getElementsByTagName('point')
    num_points = len(points)
    fs = float(data.getElementsByTagName('model')[0].getAttribute('sampleRate'))

    prev_timestamp = 0
    for beat in range(num_points):
        start_times.append(prev_timestamp)
        end_times.append(float(points[beat].getAttribute('frame')) / fs)
        prev_timestamp = float(points[beat].getAttribute('frame')) / fs

    with open(notation_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter='-')
        for row in reader:
            events.append(row[-1].replace("'", "").replace(" ", "").replace(":", ""))

        thr = events.index('pallavi')  # Get notation
        events = events[thr:]
        events = [x for x in events if len(x) < 3]  # Remove keys

    return annotations.EventData(np.array([start_times, end_times]).T, events)


def load_sections(notation_path, taala_path):
    """Load secitons

    Args:
        notation_path (str): Local path where the phrase annotation is stored.
            If `None`, returns None.
        taala_path (str): Local path where the taala annotation is stored.

    Returns:
        SectionData: section annotation for track

    """
    if notation_path is None:
        return None
    if taala_path is None:
        return None

    if not os.path.exists(notation_path):
        raise IOError("notation_path {} does not exist".format(notation_path))
    if not os.path.exists(taala_path):
        raise IOError("taala_path {} does not exist".format(taala_path))

    start_times = []
    end_times = []
    events = []

    dom = minidom.parse(taala_path)
    data = dom.getElementsByTagName('data')[0]
    points = data.getElementsByTagName('dataset')[0].getElementsByTagName('point')
    num_points = len(points)
    fs = float(data.getElementsByTagName('model')[0].getAttribute('sampleRate'))

    prev_timestamp = 0
    for beat in range(num_points):
        start_times.append(prev_timestamp)
        end_times.append(float(points[beat].getAttribute('frame')) / fs)
        prev_timestamp = float(points[beat].getAttribute('frame')) / fs

    intervals = []
    section_labels = ['pallavi', 'anupallavi', 'muktayiswaram', 'charanam', 'chittiswaram']
    with open(notation_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter='-')
        for row in reader:
            events.append(row[-1].replace("'", "").replace(" ", "").replace(":", ""))

        section_indexes = []
        for section in section_labels:
            section_indexes.append(events.index(section))
        section_indexes.append(len(end_times)-1)  # Add last index to section indexes

        for i in np.arange(1, len(section_indexes)):
            intervals.append([start_times[section_indexes[i-1]+1], end_times[section_indexes[i]-1]])

    return annotations.SectionData(np.array(intervals), section_labels)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The compmusic_carnatic_varnam dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            index=DATA.index,
            name="compmusic_carnatic_varnam",
            track_object=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_taala)
    def load_taala(self, *args, **kwargs):
        return load_taala(*args, **kwargs)

    @core.copy_docs(load_notation)
    def load_notation(self, *args, **kwargs):
        return load_notation(*args, **kwargs)

    @core.copy_docs(load_sections)
    def load_sections(self, *args, **kwargs):
        return load_sections(*args, **kwargs)
