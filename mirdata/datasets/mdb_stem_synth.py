"""MDB-stem-synth Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    MDB-stem-synth contains 230 solo stems (tracks)
    from the MedleyDB dataset spanning a variety of
    musical instruments and voices, which have been
    resynthesized to obtain a perfect f0 annotation
    using the analysis/synthesis method described in
    the referenced publication of Salamon et al.
    (ISMIR 2017).

    For more details and download info,
    please visit:
    - https://synthdatasets.weebly.com/mdb-stem-synth.html
    - https://zenodo.org/record/1481172

"""

from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np

from mirdata import io, core, annotations, download_utils

BIBTEX = """
@inproceedings{salamon2017analysis,
  title={An analysis/synthesis framework for automatic f0 annotation of multitrack datasets},
  author={Salamon, Justin and Bittner, Rachel M and Bonada, Jordi and Bosch, Juan J and G{\'o}mez Guti{\'e}rrez, Emilia and Bello, Juan Pablo},
  booktitle={International Society for Music Information Retrieval Conference},
  year={2017},
}
"""

INDEXES = {
    "default": "1.0.0",
    "test": "sample",
    "1.0.0": core.Index(
        filename="mdb_stem_synth_index_1.0.0.json",
        url="https://zenodo.org/records/14042058/files/mdb_stem_synth_index_1.0.0.json?download=1",
        checksum="4a1e8ebecfa76e6fbdaa9aa8a1ca382a",
    ),
    "sample": core.Index(filename="mdb_stem_synth_index_1.0.0_sample.json"),
}

REMOTES = {
    "mdb_stem_synth": download_utils.RemoteFileMetadata(
        filename="MDB-stem-synth.tar.gz",
        url="https://zenodo.org/records/1481172/files/MDB-stem-synth.tar.gz?download=1",
        checksum="31c1f6b4888e5fd108af91c69789a809",
        unpack_directories=["MDB-stem-synth"],
    )
}


LICENSE_INFO = """
Attribution-NonCommercial 4.0 International
"""


class Track(core.Track):
    """mdb_stem_synth Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): path to the track's audio file
        f0_path (str): path to the track's f0 annotation file
        track_id (str): track id

    Cached Properties:
        f0 (F0Data): the track's f0 annotation
        audio (Tuple[np.ndarray, float]): audio signal and sample rate
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(
            track_id,
            data_home,
            dataset_name=dataset_name,
            index=index,
            metadata=metadata,
        )

        self.audio_path = self.get_path("audio")
        self.f0_path = self.get_path("f0")

    @core.cached_property
    def f0(self) -> Optional[annotations.F0Data]:
        """The track's f0 annotation

        Returns:
            F0Data: the f0 annotation data
        """
        return load_f0(self.f0_path)

    @core.cached_property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load MDB-stem-synth audio

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - audio signal
        * float - sample rate

    """
    return librosa.load(fhandle, sr=None, mono=True)


@io.coerce_to_string_io
def load_f0(fhandle: TextIO) -> annotations.F0Data:
    """Load a MDB-stem-synth f0 annotation

    Args:
        fhandle (str or file-like): File-like object or path to f0 annotation file

    Raises:
        IOError: If f0_path does not exist

    Returns:
        F0Data: the f0 annotation data

    """
    times_frequencies = np.genfromtxt(fhandle, delimiter=",")
    return annotations.F0Data(
        times=times_frequencies[:, 0],
        time_unit="s",
        frequencies=times_frequencies[:, 1],
        frequency_unit="hz",
        voicing=(times_frequencies[:, 1] > 0).astype(np.float64),
        voicing_unit="binary",
    )


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The MDB-stem-synth dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="mdb_stem_synth",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )
