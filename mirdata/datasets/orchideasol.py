"""OrchideaSOL pitch Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    OrchidaSOL is a large collection of orchestral instruments in several playing styles. These sounds were originally recorded at Ircam in Paris (France) between 1996 and 1999, as part of a larger project named Studio On Line (SOL). One asset of OrchideaSOL is that it contains many combinations of mutes and extended playing techniques.



    OrchideaSOL is a dataset of 13265 samples, each containing a single musical note from one of 14 different instruments:
        1. Bass Tuba
        2. French Horn
        3. Trombone
        4. Trumpet in C
        5. Accordion
        6. Contrabass
        7. Violin
        8. Viola
        9. Violoncello
        10. Bassoon
        11. Clarinet in B-flat
        12. Flute
        13. Oboe
        14. Alto Saxophone


    It requires a free subscription to Ircam Forum. For more details, please visit: https://forum.ircam.fr/projects/detail/orchideasol/

"""

import csv
import json
import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
from smart_open import open

from mirdata import annotations, core, download_utils, io, jams_utils

BIBTEX = """@inproceedings{cella2020orchideasol,
    Author = {Cella, Carmine-Emanuele and Ghisi, Daniele and Lostanlen, Vincent and Lévy, Fabien and Fineberg, Joshua and Maresz, Yan},
    journal   = {CoRR},
    volume    = {abs/2007.00763},
    Title = {OrchideaSOL: a dataset of extended instrumental techniques for computer-aided orchestration},
    Month = {February},
    Title = {OrchideaSOL: an audio dataset of isolated musical notes, including mutes and extended playing techniques},
    Year = {2020}
}"""

INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="orchideasol_index_1.0.json"),
}
REMOTES = {
    "metadata": download_utils.RemoteFileMetadata(
        filename="OrchideaSOL_metadata.csv",
        url="https://zenodo.org/record/3686252/files/OrchideaSOL_metadata.csv?download=1",
        checksum="c0233a7a8f5f964f5c26e0626a28ffda",
    ),
}
DOWNLOAD_INFO = """
    To download this dataset, visit:
    https://forum.ircam.fr/projects/detail/orchideasol/
    and register to begin download.

    Once downloaded, unzip the file _OrchideaSOL_2020_release.zip
    and copy the result to:
    {}
"""

LICENSE_INFO = (
    "Creative Commons Attribution 4.0 International Public License."
)


class Track(core.Track):
    """
    """

    def __init__(
        self,
        track_id,
        data_home,
        dataset_name,
        index,
        metadata,
    ):
        super().__init__(
            track_id,
            data_home,
            dataset_name,
            index,
            metadata,
        )

        self.audio_path = self.get_path("OrchidaSOL2020")

    @property
    def instrument(self):
        return self._track_metadata.get("instrument")

    @property
    def family(self):
        return self._track_metadata.get("family")

    @property
    def technique(self):
        return self._track_metadata.get("technique")
    
    @property
    def pitch(self):
        return self._track_metadata.get("pitch")

    @property
    def title(self):
        return self._track_metadata.get("title")

    @core.cached_property
    def pitch(self) -> Optional[annotations.F0Data]:
        return load_pitch(self.pitch_path)


    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
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
            annotation_data=[(self.notes_pyin, "pyin note estimate")],
            metadata=self._track_metadata,
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a Orchidea audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    return librosa.load(fhandle, sr=None, mono=True)


@io.coerce_to_string_io
def load_pitch(fhandle: TextIO) -> annotations.F0Data:
    """load a Orchidea pitch annotation file

    Args:
        fhandle (str or file-like): str or file-like to pitch annotation file

    Raises:
        IOError: if the path doesn't exist

    Returns:
        F0Data: pitch annotation

    """
    times = []
    freqs = []
    voicing = []
    reader = csv.reader(fhandle, delimiter=",")
    for line in reader:
        times.append(float(line[0]))
        freq_val = float(line[1])
        freqs.append(freq_val)
        voicing.append(float(freq_val > 0))

    return annotations.F0Data(
        np.array(times), "s", np.array(freqs), "hz", np.array(voicing), "binary"
    )


@io.coerce_to_string_io
def load_notes(fhandle: TextIO) -> Optional[annotations.NoteData]:
    """load a note annotation file

    Args:
        fhandle (str or file-like): str or file-like to note annotation file

    Raises:
        IOError: if file doesn't exist

    Returns:
        NoteData: note annotation

    """
    intervals = []
    freqs = []
    reader = csv.reader(fhandle, delimiter=",")
    for line in reader:
        start_time = float(line[0])
        intervals.append([start_time, start_time + float(line[1])])
        freqs.append(float(line[2]))

    # if file is empty, return None
    if len(intervals) == 0:
        return None

    return annotations.NoteData(np.array(intervals), "s", np.array(freqs), "hz")


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The OrchidaSOL 2020 dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="orchideasol",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, 'OrchidaSOL_metadata.csv')
        with open(metadata_path, 'r') as fhandle:
            metadata = json.load(fhandle)

        return metadata