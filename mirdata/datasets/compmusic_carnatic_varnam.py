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

import os
import csv
import glob
import librosa

import numpy as np
from xml.dom import minidom

from mirdata import annotations, core, download_utils, io, jams_utils

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
        filename="carnatic_varnam_1.1",
        url="TODO",
        checksum="TODO",
        destination_dir=None,
    )
}

INDEXES = {
    "default": "1.1",
    "test": "1.1",
    "1.1": core.Index(filename="compmusic_carnatic_varnam_index_1.1.json"),
}

LICENSE_INFO = (
    "Creative Commons Attribution Non Commercial No Derivatives 4.0 International"
)


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
        mbid (str): musicbrainz id of the composition
        arohanam (list, str): arohanam annotation of the related raaga
        avarohanam (list, str): avarohanam annotation of the related raaga

    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        # Audio path
        self.audio_path = self.get_path("audio")

        # Annotation paths
        self.taala_path = self.get_path("taala")
        self.notation_path = self.get_path("notation")
        self.structure_path = self.get_path("structure")

    @core.cached_property
    def taala(self):
        return load_taala(self.taala_path)

    @core.cached_property
    def notation(self):
        return load_notation(self.notation_path, self.taala_path, self.structure_path)[
            0
        ]

    @core.cached_property
    def sections(self):
        return load_notation(self.notation_path, self.taala_path, self.structure_path)[
            1
        ]

    @core.cached_property
    def mbid(self):
        return load_mbid(self.notation_path)

    @core.cached_property
    def arohanam(self):
        moorchanas = load_moorchanas(self.notation_path)
        return moorchanas[0]

    @core.cached_property
    def avarohanam(self):
        moorchanas = load_moorchanas(self.notation_path)
        return moorchanas[1]

    @core.cached_property
    def artist(self):
        return self.track_id.split("_")[0]

    @core.cached_property
    def raaga(self):
        return self.track_id.split("_")[1]

    @core.cached_property
    def tonic(self):
        return self._track_metadata

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
                "tonic": self.tonic,
                "arohanam": self.arohanam,
                "avarohanam": self.avarohanam,
            },
        )


# no decorator here because of https://github.com/librosa/librosa/issues/1267
def load_audio(audio_path):
    """Load a Carnatic Varnam audio file.

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
def load_taala(fhandle):
    """Load taala annotation

    Args:
        taala_path (str): Local path where the taala annotation is stored.

    Returns:
        BeatData: taala annotation for track
    """
    # Load svl file
    dom = minidom.parse(fhandle)

    # Load data
    data = dom.getElementsByTagName("data")[0]

    # Store points and calculate total length
    points = data.getElementsByTagName("dataset")[0].getElementsByTagName("point")
    num_points = len(points)

    # Parse sampling frequency
    fs = float(data.getElementsByTagName("model")[0].getAttribute("sampleRate"))

    beat_times = []
    beat_positions = []
    for beat in range(num_points):
        beat_times.append(float(points[beat].getAttribute("frame")) / fs)
        beat_positions.append(0)

    return annotations.BeatData(
        np.array(beat_times), "s", np.array(beat_positions), "global_index"
    )


# no decorator here because we need three paths
def load_notation(note_path, taala_path, structure_path):
    """Load notation (notes)

    Args:
        notation_path (str): Local path where the phrase annotation is stored.
            If `None`, returns None.
        taala_path (str): Local path where the taala annotation is stored.
            If `None`, returns None.

    Returns:
        EventData: melodic notation for track

    """
    if note_path is None or taala_path is None:
        return None

    start_times = []
    end_times = []
    events = []

    dom = minidom.parse(taala_path)
    data = dom.getElementsByTagName("data")[0]
    points = data.getElementsByTagName("dataset")[0].getElementsByTagName("point")
    num_points = len(points)
    fs = float(data.getElementsByTagName("model")[0].getAttribute("sampleRate"))

    prev_timestamp = float(points[0].getAttribute("frame")) / fs
    for beat in range(1, num_points):
        start_times.append(prev_timestamp)
        end_times.append(float(points[beat].getAttribute("frame")) / fs)
        prev_timestamp = float(points[beat].getAttribute("frame")) / fs
    start_times.append(prev_timestamp)
    end_times.append(prev_timestamp + (end_times[-1] - start_times[-2]))

    # Getting structure
    with open(structure_path, "r") as fhandle:
        structure = []
        reader = csv.reader(fhandle, delimiter=":")
        for row in reader:
            if len(row) > 1:
                ky = row[0].replace("'", "").replace(" ", "").replace(":", "")
                vl = row[1].replace("'", "").replace(" ", "").replace(":", "")
                if vl and len(vl) < 2:
                    structure.append((ky, int(vl)))

    print("...")
    print(structure)
    print("...")

    # Getting notation
    with open(note_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter="-")
        for row in reader:
            events.append(row[-1].replace("'", "").replace(" ", "").replace(":", ""))

        notation_dict = {}
        section_dict = {
            events.index(x): x for x in list(np.unique([x[0] for x in structure]))
        }
        start_idxs = sorted(section_dict.keys())
        end_idxs = sorted(section_dict.keys())[1:] + [len(events)]
        for start, end in zip(start_idxs, end_idxs):
            notation_dict[section_dict[start]] = events[start + 1 : end]

    # Putting all together
    print("get in there!!")
    print(section_dict)
    print(events)
    events = []
    intervals = []
    section_labels = []
    for section in structure:
        not_per_sec = notation_dict[section[0]]
        section_start = start_times[len(events)]
        if section[1] == 2:
            for x in np.arange(len(not_per_sec), step=2):
                # notes = [not_per_sec[x], not_per_sec[x+1]]
                notes = not_per_sec[x] + not_per_sec[x + 1]
                events.append(notes)
        if section[1] == 4:
            for x in np.arange(len(not_per_sec), step=4):
                # notes = [not_per_sec[x], not_per_sec[x+1], not_per_sec[x+2], not_per_sec[x+3]]
                notes = (
                    not_per_sec[x]
                    + not_per_sec[x + 1]
                    + not_per_sec[x + 2]
                    + not_per_sec[x + 3]
                )
                events.append(notes)
        section_end = end_times[len(events) - 1]
        intervals.append([section_start, section_end])
        section_labels.append(section[0])

    notes = annotations.EventData(
        np.array([start_times, end_times]).T, "s", events, "open"
    )
    sections = annotations.SectionData(np.array(intervals), "s", section_labels, "open")

    print(note_path)
    print("---", notes)

    return notes, sections


@io.coerce_to_string_io
def load_mbid(fhandle):
    """Load musicbrainz id

    Args:
        fhandle (str or file-like): Local path where the annotation is stored.
            If `None`, returns None.

    Returns:
        string: musicbrainz id for the composition

    """
    reader = csv.reader(fhandle, delimiter=":")
    for row in reader:
        if row[0] == "mbid":
            return row[-1].replace("'", "").replace(" ", "")


@io.coerce_to_string_io
def load_moorchanas(fhandle):
    """Load arohanam and avarohanam annotations

    Args:
        fhandle (str or file-like): Local path where moorchana annotation is stored.
            If `None`, returns None.

    Returns:
        (list, string): section annotation for track

    """
    notes = []
    reader = csv.reader(fhandle, delimiter="-")
    for row in reader:
        if row[0] == "pallavi:":
            break
        notes.append(str(row[-1].replace(" ", "")))

    arohanam_ind = (
        notes.index("arohana:") + 1
    )  # Get left boundary of arohanam notations
    avarohanam_ind = (
        notes.index("avarohana:") + 1
    )  # Get left boundary of avarohanam notations

    arohanam = notes[arohanam_ind : avarohanam_ind - 1]  # Get arohanam
    avarohanam = notes[avarohanam_ind:]  # Get avarohanam

    return [arohanam, avarohanam]


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The compmusic_carnatic_varnam dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="compmusic_carnatic_varnam",
            track_class=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            indexes=INDEXES,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        """Load tonic

        Args:
            fhandle (str or file-like): Local path where tonic annotations are stored.
                If `None`, returns None.
            track_id (str): Track ID to get the artist name

        Returns:
            (float): tonic

        """
        data_folder = self.remotes["all"].filename
        tonics_dict = {}
        tonics_path = os.path.join(
            self.data_home,
            data_folder,
            "Notations_Annotations",
            "annotations",
            "tonics.yaml",
        )
        with open(tonics_path, "r") as f:
            reader = csv.reader(f, delimiter=":")
            for line in reader:
                tonics_dict[line[0]] = float(line[1])

        taalas_path = os.path.join(
            self.data_home,
            data_folder,
            "Notations_Annotations",
            "annotations",
            "taalas",
        )
        out_tonic = {}
        for taala in glob.glob(os.path.join(taalas_path, "*/")):
            for track in glob.glob(os.path.join(taala, "*.svl")):
                taala = taala.split("/")[-2]
                artist = track.split("/")[-1].replace(".svl", "")
                idx = artist + "_" + taala
                out_tonic[idx] = tonics_dict[artist]
        return out_tonic

    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    def load_taala(self, *args, **kwargs):
        return load_taala(*args, **kwargs)

    def load_notation(self, *args, **kwargs):
        return load_notation(*args, **kwargs)

    def load_mbid(self, *args, **kwargs):
        return load_mbid(*args, **kwargs)

    def load_moorchanas(self, *args, **kwargs):
        return load_moorchanas(*args, **kwargs)
