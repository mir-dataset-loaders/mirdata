# -*- coding: utf-8 -*-
"""
cante100 Loader

.. admonition:: Dataset Info
    :class: dropdown

    The cante100 dataset contains 100 tracks taken from the COFLA corpus. We defined 10 style
    families of which 10 tracks each are included. Apart from the style family, we manually
    annotated the sections of the track in which the vocals are present. In addition, we
    provide a number of low-level descriptors and the fundamental frequency corresponding to
    the predominant melody for each track. The meta-information includes editoral meta-data
    and the musicBrainz ID.

    Total tracks: 100

    cante100 audio is only available upon request. To download the audio request access in
    this link: https://zenodo.org/record/1324183. Then
    unzip the audio into the cante100 general dataset folder for the rest of annotations
    and files.

    Audio specifications:

    - Sampling frequency: 44.1 kHz
    - Bit-depth: 16 bit
    - Audio format: .mp3

    cante100 dataset has spectrogram available, in csv format. spectrogram is available to download
    without request needed, so at first instance, cante100 loader uses the spectrogram of the tracks.

    The available annotations are:

    - F0 (predominant melody)
    - Automatic transcription of notes (of singing voice)

    CANTE100 LICENSE (COPIED FROM ZENODO PAGE)

    .. code-block:: latex

        The provided datasets are offered free of charge for internal non-commercial use.
        We do not grant any rights for redistribution or modification. All data collections were gathered
        by the COFLA team.
        © COFLA 2015. All rights reserved.

    For more details, please visit: http://www.cofla-project.com/?page_id=134

"""
import csv
import os
import logging
import xml.etree.ElementTree as ET

import librosa
import numpy as np

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core
from mirdata import annotations


BIBTEX = """@dataset{nadine_kroher_2018_1322542,
  author       = {Nadine Kroher and
                  José Miguel Díaz-Báñez and
                  Joaquin Mora and
                  Emilia Gómez},
  title        = {cante100 Metadata},
  month        = jul,
  year         = 2018,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.1322542},
  url          = {https://doi.org/10.5281/zenodo.1322542}
},
@dataset{nadine_kroher_2018_1324183,
  author       = {Nadine Kroher and
                  José Miguel Díaz-Báñez and
                  Joaquin Mora and
                  Emilia Gómez},
  title        = {cante100 Audio},
  month        = jul,
  year         = 2018,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.1324183},
  url          = {https://doi.org/10.5281/zenodo.1324183}
}
"""


REMOTES = {
    "spectrogram": download_utils.RemoteFileMetadata(
        filename="cante100_spectrum.zip",
        url="https://zenodo.org/record/1322542/files/cante100_spectrum.zip?download=1",
        checksum="0b81fe0fd7ab2c1adc1ad789edb12981",  # the md5 checksum
        destination_dir="cante100_spectrum",  # relative path for where to unzip the data, or None
    ),
    "melody": download_utils.RemoteFileMetadata(
        filename="cante100midi_f0.zip",
        url="https://zenodo.org/record/1322542/files/cante100midi_f0.zip?download=1",
        checksum="cce543b5125eda5a984347b55fdcd5e8",  # the md5 checksum
        destination_dir="cante100midi_f0",  # relative path for where to unzip the data, or None
    ),
    "notes": download_utils.RemoteFileMetadata(
        filename="cante100_automaticTranscription.zip",
        url="https://zenodo.org/record/1322542/files/cante100_automaticTranscription.zip?download=1",
        checksum="47fea64c744f9fe678ae5642a8f0ee8e",  # the md5 checksum
        destination_dir="cante100_automaticTranscription",  # relative path for where to unzip the data, or None
    ),
    "metadata": download_utils.RemoteFileMetadata(
        filename="cante100Meta.xml",
        url="https://zenodo.org/record/1322542/files/cante100Meta.xml?download=1",
        checksum="6cce186ce77a06541cdb9f0a671afb46",  # the md5 checksum
        destination_dir=None,  # relative path for where to unzip the data, or None
    ),
    "README": download_utils.RemoteFileMetadata(
        filename="cante100_README.txt",
        url="https://zenodo.org/record/1322542/files/cante100_README.txt?download=1",
        checksum="184209b7e7d816fa603f0c7f481c0aae",  # the md5 checksum
        destination_dir=None,  # relative path for where to unzip the data, or None
    ),
}


DOWNLOAD_INFO = """
        This loader is designed to load the spectrum, as it is available for download.
        However, the loader supports audio as well. Unfortunately the audio files of the 
        cante100 dataset are not available for free download, but upon request. However, 
        you can request de audio in both links here: 
        ==> http://www.cofla-project.com/?page_id=208
        ==> https://zenodo.org/record/1324183
        Then, locate the downloaded the cante100audio folder like this:
            > cante100/
                > cante100_spectrum/
                ... (rest of the annotation folders)
                > cante100audio/
        Remember to locate the cante100 folder to {}
"""

LICENSE_INFO = """
The provided datasets are offered free of charge for internal non-commercial use.
We do not grant any rights for redistribution or modification. All data collections
were gathered by the COFLA team. COFLA 2015. All rights reserved.
"""


def _load_metadata(data_home):
    metadata_path = os.path.join(data_home, "cante100Meta.xml")
    if not os.path.exists(metadata_path):
        logging.info(
            "Metadata file {} not found.".format(metadata_path)
            + "You can download the metadata file for cante100 "
            + "by running cante100.download()"
        )
        return None

    tree = ET.parse(metadata_path)
    root = tree.getroot()

    # ids
    indexes = []
    for child in root:
        index = child.attrib.get("id")
        if len(index) == 1:
            index = "00" + index
            indexes.append(index)
            continue
        if len(index) == 2:
            index = "0" + index
            indexes.append(index)
            continue
        else:
            indexes.append(index)

    # musicBrainzID
    identifiers = []
    for ident in root.iter("musicBrainzID"):
        identifiers.append(ident.text)

    # artist
    artists = []
    for artist in root.iter("artist"):
        artists.append(artist.text)

    # titles
    titles = []
    for title in root.iter("title"):
        titles.append(title.text)

    # releases
    releases = []
    for release in root.iter("anthology"):
        releases.append(release.text)

    # duration
    durations = []
    minutes = []
    for minute in root.iter("duration_m"):
        minutes.append(float(minute.text) * 60)
    seconds = []
    for second in root.iter("duration_s"):
        seconds.append(float(second.text))
    for i in np.arange(len(minutes)):
        durations.append(minutes[i] + seconds[i])

    metadata = dict()
    metadata["data_home"] = data_home
    for i, j in zip(indexes, range(len(artists))):
        metadata[i] = {
            "musicBrainzID": identifiers[j],
            "artist": artists[j],
            "title": titles[j],
            "release": releases[j],
            "duration": durations[j],
        }

    return metadata


DATA = core.LargeData("cante100_index.json", _load_metadata)


class Track(core.Track):
    """cante100 track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/cante100`

    Attributes:
        track_id (str): track id
        identifier (str): musicbrainz id of the track
        artist (str): performing artists
        title (str): title of the track song
        release (str): release where the track can be found
        duration (str): duration in seconds of the track

    Cached Properties:
        melody (F0Data): annotated melody
        notes (NoteData): annotated notes

    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index["tracks"]:
            raise ValueError("{} is not a valid track ID in Example".format(track_id))

        self.track_id = track_id

        self._data_home = data_home

        self._track_paths = DATA.index["tracks"][track_id]
        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])
        self.spectrogram_path = os.path.join(
            self._data_home, self._track_paths["spectrum"][0]
        )
        self.f0_path = os.path.join(self._data_home, self._track_paths["f0"][0])
        self.notes_path = os.path.join(self._data_home, self._track_paths["notes"][0])

        metadata = DATA.metadata(data_home=data_home)
        if metadata is not None and track_id in metadata:
            self._track_metadata = metadata[track_id]
        else:
            self._track_metadata = {
                "musicBrainzID": None,
                "artist": None,
                "title": None,
                "release": None,
                "duration": None,
            }

        self.identifier = self._track_metadata["musicBrainzID"]
        self.artist = self._track_metadata["artist"]
        self.title = self._track_metadata["title"]
        self.release = self._track_metadata["release"]
        self.duration = self._track_metadata["duration"]

    @property
    def audio(self):
        """The track's audio

        Returns:
           * np.ndarray - audio signal
           * float - sample rate

        """
        return load_audio(self.audio_path)

    @property
    def spectrogram(self):
        """spectrogram of The track's audio

        Returns:
            (np.ndarray): spectrogram
        """
        return load_spectrogram(self.spectrogram_path)

    @core.cached_property
    def melody(self):
        return load_melody(self.f0_path)

    @core.cached_property
    def notes(self):
        return load_notes(self.notes_path)

    def to_jams(self):
        """Get the track's data in jams format

        Returns:
            jams.JAMS: the track's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            spectrogram_path=self.spectrogram_path,
            f0_data=[(self.melody, "pitch_contour")],
            note_data=[(self.notes, "note_hz")],
            metadata=self._track_metadata,
        )


def load_spectrogram(spectrogram_path):
    """Load a cante100 dataset spectrogram file.

    Args:
        spectrogram_path (str): path to audio file

    Returns:
        np.ndarray: spectrogram

    """
    if not os.path.exists(spectrogram_path):
        raise IOError("spectrogram_path {} does not exist".format(spectrogram_path))
    parsed_spectrogram = np.genfromtxt(spectrogram_path, delimiter=" ")
    spectrogram = parsed_spectrogram.astype(np.float)

    return spectrogram


def load_audio(audio_path):
    """Load a cante100 audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    audio, sr = librosa.load(audio_path, sr=22050, mono=False)
    return audio, sr


def load_melody(f0_path):
    """Load cante100 f0 annotations

    Args:
        f0_path (str): path to audio file

    Returns:
        F0Data: predominant melody

    """
    if not os.path.exists(f0_path):
        raise IOError("f0_path {} does not exist".format(f0_path))

    times = []
    freqs = []
    with open(f0_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter=",")
        for line in reader:
            times.append(float(line[0]))
            freqs.append(float(line[1]))

    times = np.array(times)
    freqs = np.array(freqs)
    confidence = (freqs > 0).astype(float)

    return annotations.F0Data(times, freqs, confidence)


def load_notes(notes_path):
    """Load note data from the annotation files

    Args:
        notes_path (str): path to notes file

    Returns:
        NoteData: note annotations

    """
    if not os.path.exists(notes_path):
        raise IOError("notes_path {} does not exist".format(notes_path))

    intervals = []
    pitches = []
    confidence = []
    with open(notes_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter=",")
        for line in reader:
            intervals.append([line[0], float(line[0]) + float(line[1])])
            # Convert midi value to frequency
            pitches.append((440 / 32) * (2 ** ((int(line[2]) - 9) / 12)))
            confidence.append(1.0)

    return annotations.NoteData(
        np.array(intervals, dtype="float"),
        np.array(pitches, dtype="float"),
        np.array(confidence, dtype="float"),
    )


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The cante100 dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            index=DATA.index,
            name="cante100",
            track_object=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_spectrogram)
    def load_spectrogram(self, *args, **kwargs):
        return load_spectrogram(*args, **kwargs)

    @core.copy_docs(load_melody)
    def load_melody(self, *args, **kwargs):
        return load_melody(*args, **kwargs)

    @core.copy_docs(load_notes)
    def load_notes(self, *args, **kwargs):
        return load_notes(*args, **kwargs)
