"""
FMAK / FMAKv2 Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    **FMAK**

    FMAK is an expert-labeled dataset for the evaluation of key detection containing
    340 hours (5489 songs) of song-level key and mode annotations, spread across 17 genres.

    The curation and annotations of FMAK were created by Stella Wong (co-author of STONE)
    and Gandalf Hernandez. The dataset was first presented as an ISMIR Late-Breaking/Demo
    in 2023, and later released and used in the paper
    *STONE: Self-supervised Tonality Estimator* (ISMIR 2024).

    DOI: https://doi.org/10.5281/zenodo.10719860

    For each song, FMAK provides:

    - FMA track id (6 digits)
    - Spotify URI (when available)
    - Key and mode

    The modes are provided both as strings and numbers:
        "Major" <-> 1, "minor" <-> 0

    Similarly, for the keys:
        "C" <-> 0, "C#" <-> 1, etc.

    All audio comes from the FMA dataset
    (Michael Defferrard, Kirell Benzi, Pierre Vandergheynst, and Xavier Bresson).

    **FMAKv2**

    FMAKv2 is a derivative work of FMAK, released and used in the ISMIR 2024 paper
    *STONE: Self-supervised Tonality Estimator*. The difference between FMAK and FMAKv2
    is a modification of around 200 annotations. All other annotations remain unchanged
    from FMAK. FMA track id and Spotify URI remain identical.

    Authors of FMAK did not verify the modifications of FMAKv2 and should not be held
    liable for potential mislabelings.

    The audio is identical to FMAK and can be obtained from the FMA dataset.

    DOI: https://doi.org/10.5281/zenodo.12759100

    **Versions in mirdata**

    - ``"1.0"``: FMAK annotations (``fma_keys_metadata.csv``)
    - ``"2.0"`` (default): FMAKv2 annotations (``fmakv2_metadata.csv``)

    Both versions share the same audio files and track ids. To download only the
    annotations, use ``partial_download=["metadata"]`` for ``"1.0"`` and
    ``partial_download=["metadata_v2"]`` for ``"2.0"``.

    **License**

    The FMAK and FMAKv2 annotations are released under the Creative Commons
    Attribution 4.0 International license (CC BY 4.0). The dataset is intended
    for research purposes.

    The dataset authors do not hold the copyright on the audio. It is distributed
    by the FMA dataset under the license chosen by each artist, which varies
    per track.

    **Links**

    - FMAK: https://zenodo.org/records/10719860
    - FMAKv2: https://zenodo.org/records/12759100
    - FMAK code and annotations: https://github.com/stellaywong/fma_keys
    - FMA dataset: https://github.com/mdeff/fma

    **Citations**

    If you use FMAKv2, please cite both papers below, as requested by its authors.
    If you use FMAK (version ``"1.0"``), please cite the FMAK paper:

    .. code-block:: bibtex

        @article{kong2024stone,
          title={STONE: Self-supervised Tonality Estimator},
          author={Kong, Yuexuan and Lostanlen, Vincent and Meseguer-Brocal, Gabriel and Wong, Stella and Lagrange, Mathieu and Hennequin, Romain},
          journal={Proceedings of International Society for Music Information Retrieval Conference (ISMIR 2024)},
          year={2024}
        }

        @inproceedings{wong2023fmak,
          title={FMAK: A DATASET OF KEY AND MODE ANNOTATIONS FOR THE FREE MUSIC ARCHIVE--EXTENDED ABSTRACT},
          author={Wong, Stella and Hernandez, Gandalf},
          booktitle={International Society for Music Information Retrieval Late-Breaking/Demo Session (ISMIR-LBD)},
          year={2023}
        }
"""

import csv
import os
from typing import Optional, Tuple

import librosa
import numpy as np
from smart_open import open

from mirdata import core, download_utils


BIBTEX = """
@inproceedings{
  wong_fma_keys,
  title = {FMAK: A Dataset of Key and Mode Annotations for the Free Music Archive -- Extended Abstract},
  author = {Wong, Stella and Hernandez, Gandalf},
  booktitle = {International Society for Music Information Retrieval Late-Breaking/Demo Session (ISMIR-LBD)},
  year = {2023}
}
@article{kong2024stone,
  title={STONE: Self-supervised Tonality Estimator},
  author={Kong, Yuexuan and Lostanlen, Vincent and Meseguer-Brocal, Gabriel and Wong, Stella and Lagrange, Mathieu and Hennequin, Romain},
  journal={Proceedings of International Society for Music Information Retrieval Conference (ISMIR 2024)},
  year={2024}
}
"""

LICENSE_INFO = "Creative Commons Attribution 4.0 International"

# Both versions share the audio; each index only downloads its own metadata file.
AUDIO_REMOTES = [
    "tracks-000-019",
    "tracks-020-039",
    "tracks-040-049",
    "tracks-050-059",
    "tracks-060-069",
    "tracks-070-079",
    "tracks-080-089",
    "tracks-090-099",
    "tracks-100-109",
    "tracks-110-124",
]

INDEXES = {
    "default": "2.0",
    "test": "sample",
    "1.0": core.Index(
        filename="fma_keys_index_1.0.json",
        url="https://zenodo.org/records/16757314/files/fma_keys_index_1.0.json?download=1",
        checksum="6c905f1c0d1caef11643b67cfe80ddf4",
        partial_download=AUDIO_REMOTES + ["metadata"],
    ),
    "2.0": core.Index(
        filename="fmakv2_index_1.0.json",
        url="https://zenodo.org/records/17182864/files/fmakv2_index_1.0.json?download=1",
        checksum="abebede26962c58fd8b78f4b6873d192",
        partial_download=AUDIO_REMOTES + ["metadata_v2"],
    ),
    "sample": core.Index(filename="fma_keys_index_1.0_sample.json"),
}

REMOTES = {
    "metadata": download_utils.RemoteFileMetadata(
        filename="fma_keys_metadata.csv",
        url="https://zenodo.org/records/10719860/files/fma_keys_metadata.csv?download=1",
        checksum="d80a03bc8659edc60e335bd7f6bdf12a",
    ),
    # Saved as fmakv2_metadata.csv (not the Zenodo name fmakv2.csv) because that is
    # the path the published v2 index expects; changing it breaks validate().
    "metadata_v2": download_utils.RemoteFileMetadata(
        filename="fmakv2_metadata.csv",
        url="https://zenodo.org/records/12759100/files/fmakv2.csv?download=1",
        checksum="3b2d16784ffbda850c8ddf0519478bfd",
    ),
    "tracks-000-019": download_utils.RemoteFileMetadata(
        filename="000-019.zip",
        url="https://zenodo.org/records/10719860/files/000-019.zip?download=1",
        checksum="b86f6414820c1422b2c6cdf87be1ef3a",
    ),
    "tracks-020-039": download_utils.RemoteFileMetadata(
        filename="020-039.zip",
        url="https://zenodo.org/records/10719860/files/020-039.zip?download=1",
        checksum="a2da8377fdbc1d3a1f54dd60aa7b8f9b",
    ),
    "tracks-040-049": download_utils.RemoteFileMetadata(
        filename="040-049.zip",
        url="https://zenodo.org/records/10719860/files/040-049.zip?download=1",
        checksum="d70babe5f66bdf3e821c42a8b8aafb9b",
    ),
    "tracks-050-059": download_utils.RemoteFileMetadata(
        filename="050-059.zip",
        url="https://zenodo.org/records/10719860/files/050-059.zip?download=1",
        checksum="f53fcba704fce27e5c7f3ec2532dcb44",
    ),
    "tracks-060-069": download_utils.RemoteFileMetadata(
        filename="060-069.zip",
        url="https://zenodo.org/records/10719860/files/060-069.zip?download=1",
        checksum="1520f067d7caaf0813780ff69bc4ba85",
    ),
    "tracks-070-079": download_utils.RemoteFileMetadata(
        filename="070-079.zip",
        url="https://zenodo.org/records/10719860/files/070-079.zip?download=1",
        checksum="186643746fcb1f4722a28d3eb9c6b99c",
    ),
    "tracks-080-089": download_utils.RemoteFileMetadata(
        filename="080-089.zip",
        url="https://zenodo.org/records/10719860/files/080-089.zip?download=1",
        checksum="8cf882609fc2f301621c2e9f9da03214",
    ),
    "tracks-090-099": download_utils.RemoteFileMetadata(
        filename="090-099.zip",
        url="https://zenodo.org/records/10719860/files/090-099.zip?download=1",
        checksum="84f0f036e3778ffd97c10b591f803d06",
    ),
    "tracks-100-109": download_utils.RemoteFileMetadata(
        filename="100-109.zip",
        url="https://zenodo.org/records/10719860/files/100-109.zip?download=1",
        checksum="4a307f019d3354064814f05d1dffa1e2",
    ),
    "tracks-110-124": download_utils.RemoteFileMetadata(
        filename="110-124.zip",
        url="https://zenodo.org/records/10719860/files/110-124.zip?download=1",
        checksum="88d7dbcca82189ed75b7baa5aa132fc1",
    ),
}

METADATA_FILENAMES = {
    "1.0": "fma_keys_metadata.csv",
    "2.0": "fmakv2_metadata.csv",
    "sample": "fma_keys_metadata.csv",
}


KEY_MAP = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "Bb": 10,
    "B": 11,
}
MODE_MAP = {"minor": 0, "Major": 1}


class Track(core.Track):
    """FMA Keys Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        spotify_uri (str): Spotify URI if available
        key (str): key of the track (C, C#, etc)
        mode (str): mode of the track (Major, minor)
        key_number (int): numeric key of the track (0-11)
        mode_number (int): numeric mode of the track (0 for minor, 1 for Major)
        audio_path (str): path to the track's audio file
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)
        self.audio_path = self.get_path("audio")

    @property
    def spotify_uri(self):
        return self._track_metadata.get("spotify_uri")

    @property
    def key(self):
        return self._track_metadata.get("key")

    @property
    def mode(self):
        return self._track_metadata.get("mode")

    @property
    def key_number(self):
        return self._track_metadata.get("key_number")

    @property
    def mode_number(self):
        return self._track_metadata.get("mode_number")

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_audio(self.audio_path)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The FMA Keys dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="fma_keys",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    def _track_to_dict(self, t):
        key_and_mode = t["key_and_mode"].split(" ")

        return {
            "spotify_uri": t["spotify_uri"],
            "key": key_and_mode[0],
            "mode": key_and_mode[1],
            "key_number": KEY_MAP[key_and_mode[0]],
            "mode_number": MODE_MAP[key_and_mode[1]],
        }

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, METADATA_FILENAMES[self.version])

        try:
            with open(metadata_path) as f:
                return {
                    t["track_id"]: self._track_to_dict(t) for t in csv.DictReader(f)
                }
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Metadata not found at {metadata_path}. Did you run .download()?"
            )


# no decorator here because of https://github.com/librosa/librosa/issues/1267
def load_audio(path: str) -> Tuple[np.ndarray, float]:
    """Load fma keys audio

    Args:
        path(str): Path to audio file

    Returns:
        * np.ndarray - audio signal
        * float - sample rate
    """
    return librosa.load(path, sr=None, mono=True)
