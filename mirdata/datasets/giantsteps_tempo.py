# -*- coding: utf-8 -*-
"""giantsteps_tempo Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    GiantSteps tempo + genre is a collection of annotations for 664 2min(1) audio previews from
    www.beatport.com, created by Richard Vogl <richard.vogl@tuwien.ac.at> and 
    Peter Knees <peter.knees@tuwien.ac.at>

    references:

    .. [giantsteps_tempo_cit_1] Peter Knees, Ángel Faraldo, Perfecto Herrera, Richard Vogl,
        Sebastian Böck, Florian Hörschläger, Mickael Le Goff: "Two data
        sets for tempo estimation and key detection in electronic dance
        music annotated from user corrections", Proc. of the 16th
        Conference of the International Society for Music Information
        Retrieval (ISMIR'15), Oct. 2015, Malaga, Spain.

    .. [giantsteps_tempo_cit_2] Hendrik Schreiber, Meinard Müller: "A Crowdsourced Experiment
        for Tempo Estimation of Electronic Dance Music", Proc. of the
        19th Conference of the International Society for Music
        Information Retrieval (ISMIR'18), Sept. 2018, Paris, France.

    The audio files (664 files, size ~1gb) can be downloaded from http://www.beatport.com/
    using the bash script:

    https://github.com/GiantSteps/giantsteps-tempo-dataset/blob/master/audio_dl.sh

    To download the files manually use links of the following form:
    http://geo-samples.beatport.com/lofi/<name of mp3 file>
    e.g.:
    http://geo-samples.beatport.com/lofi/5377710.LOFI.mp3

    To convert the audio files to .wav use the script found at 
    https://github.com/GiantSteps/giantsteps-tempo-dataset/blob/master/convert_audio.sh and run:

    .. code-block:: bash

        ./convert_audio.sh

    To retrieve the genre information, the JSON contained within the website was parsed.
    The tempo annotation was extracted from forum entries of people correcting the bpm values (i.e. manual annotation of tempo).
    For more information please refer to the publication [giantsteps_tempo_cit_1]_.

    [giantsteps_tempo_cit_2]_ found some files without tempo. There are:

    .. code-block:: bash

        3041381.LOFI.mp3
        3041383.LOFI.mp3
        1327052.LOFI.mp3

    Their v2 tempo is denoted as 0.0 in tempo and mirex and has no annotation in the JAMS format.

    Most of the audio files are 120 seconds long. Exceptions are:

    .. code-block:: bash

        name              length (sec)
        906760.LOFI.mp3   62
        1327052.LOFI.mp3  70
        4416506.LOFI.mp3  80
        1855660.LOFI.mp3  119
        3419452.LOFI.mp3  119
        3577631.LOFI.mp3  119

"""
import os

import jams
import librosa
import numpy as np

from mirdata import download_utils
from mirdata import core
from mirdata import annotations


BIBTEX = """@inproceedings{knees2015two,
  title={Two data sets for tempo estimation and key detection in electronic dance music annotated from user corrections},
  author={Knees, Peter and Faraldo P{\'e}rez, {\'A}ngel and Boyer, Herrera and Vogl, Richard and B{\"o}ck, Sebastian and H{\"o}rschl{\"a}ger, Florian and Le Goff, Mickael and others},
  booktitle={Proceedings of the 16th International Society for Music Information Retrieval Conference (ISMIR); 2015 Oct 26-30; M{\'a}laga, Spain.[M{\'a}laga]: International Society for Music Information Retrieval, 2015. p. 364-70.},
  year={2015},
  organization={International Society for Music Information Retrieval (ISMIR)},
}
@inproceedings{SchreiberM18a_Tempo_ISMIR,
  author={Hendrik Schreiber and Meinard M{\"u}ller},
  title={A Crowdsourced Experiment for Tempo Estimation of Electronic Dance Music},
  booktitle={Proceedings of the International Conference on Music Information Retrieval ({ISMIR})},
  address={Paris, France},
  year={2018},
  url-pdf={http://www.tagtraum.com/download/2018_schreiber_tempo_giantsteps.pdf},
}"""

DATA = core.LargeData("giantsteps_tempo_index.json")

REMOTES = {
    "annotations": download_utils.RemoteFileMetadata(
        filename="giantsteps-tempo-dataset-0b7d47ba8cae59d3535a02e3db69e2cf6d0af5bb.zip",
        url="https://github.com/GiantSteps/giantsteps-tempo-dataset/archive/0b7d47ba8cae59d3535a02e3db69e2cf6d0af5bb.zip",
        checksum="8fdafbaf505fe3f293bd912c92b72ac8",
        destination_dir="",
    )
}
DOWNLOAD_INFO = """
    Unfortunately the audio files of the Giant Steps Tempo dataset are not available
    for download. If you have the Giant Steps audio dataset, place the contents into
    a folder called GiantSteps_tempo with the following structure:
        > GiantSteps_tempo/
            > giantsteps-tempo-dataset-0b7d47ba8cae59d3535a02e3db69e2cf6d0af5bb/
            > audio/
    and copy the folder to {}
"""

LICENSE_INFO = "Creative Commons Attribution Share Alike 4.0 International."


class Track(core.Track):
    """giantsteps_tempo track class

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): track audio path
        title (str): title of the track
        track_id (str): track id
        annotation_v1_path (str): track annotation v1 path
        annotation_v2_path (str): track annotation v2 path

    Cached Properties:
        genre (dict): Human-labeled metadata annotation
        tempo (list): List of annotations.TempoData, ordered by confidence
        tempo_v2 (list): List of annotations.TempoData for version 2, ordered by confidence

    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index["tracks"]:
            raise ValueError(
                "{} is not a valid track ID in giantsteps_tempo".format(track_id)
            )

        self.track_id = track_id

        self._data_home = data_home
        self._track_paths = DATA.index["tracks"][track_id]
        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])
        self.annotation_v1_path = os.path.join(
            self._data_home, self._track_paths["annotation_v1"][0]
        )
        self.annotation_v2_path = os.path.join(
            self._data_home, self._track_paths["annotation_v2"][0]
        )

        self.title = self.audio_path.replace(".mp3", "").split("/")[-1].split(".")[0]

    @core.cached_property
    def genre(self):
        return load_genre(self.annotation_v1_path)

    @core.cached_property
    def tempo(self):
        return load_tempo(self.annotation_v1_path)

    @core.cached_property
    def tempo_v2(self):
        return load_tempo(self.annotation_v2_path)

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
        return jams.load(self.annotation_v1_path)

    def to_jams_v2(self):
        """Get the track's data in jams format

        Returns:
            jams.JAMS: the track's data in jams format

        """
        return jams.load(self.annotation_v2_path)


def load_audio(audio_path):
    """Load a giantsteps_tempo audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=None, mono=True)


def load_genre(path):
    """Load genre data from a file

    Args:
        path (str): path to metadata annotation file

    Returns:
        str: loaded genre data

    """
    if path is None:
        return None

    with open(path) as json_file:
        annotation = jams.load(json_file)

    return annotation.search(namespace="tag_open")[0]["data"][0].value


def load_tempo(tempo_path):
    """Load giantsteps_tempo tempo data from a file ordered by confidence

    Args:
        tempo_path (str): path to tempo annotation file

    Returns:
        list: list of annotations.TempoData

    """
    if tempo_path is None:
        return None

    if not os.path.exists(tempo_path):
        raise IOError("tempo_path {} does not exist".format(tempo_path))

    with open(tempo_path) as json_file:
        annotation = jams.load(json_file)

    tempo = annotation.search(namespace="tempo")[0]["data"]

    return annotations.TempoData(
        np.array([[t.time for t in tempo], [t.time + t.duration for t in tempo]]).T,
        np.array([t.value for t in tempo]),
        np.array([t.confidence for t in tempo]),
    )


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The giantsteps_tempo dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            index=DATA.index,
            name="giantsteps_tempo",
            track_object=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_genre)
    def load_genre(self, *args, **kwargs):
        return load_genre(*args, **kwargs)

    @core.copy_docs(load_tempo)
    def load_tempo(self, *args, **kwargs):
        return load_tempo(*args, **kwargs)
