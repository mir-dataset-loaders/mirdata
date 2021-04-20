"""slakh Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    The Synthesized Lakh (Slakh) Dataset is a dataset of multi-track audio and aligned 
    MIDI for music source separation and multi-instrument automatic transcription. 
    Individual MIDI tracks are synthesized from the Lakh MIDI Dataset v0.1 using 
    professional-grade sample-based virtual instruments, and the resulting audio is 
    mixed together to make musical mixtures. 
    
    The original release of Slakh, called Slakh2100, 
    contains 2100 automatically mixed tracks and accompanying, aligned MIDI files, 
    synthesized from 187 instrument patches categorized into 34 classes, totaling 
    145 hours of mixture data.

    This loader supports two versions of Slakh:
    - Slakh2100-redux: a deduplicated version of Slack2100 containing 1710 multitracks
    - baby-slakh: a mini version with 16k wav audio and only the first 20 tracks

    For more information see http://www.slakh.com/

"""
import csv
import logging
import json
import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np

from mirdata import download_utils, jams_utils, core, annotations

BIBTEX = """
@inproceedings{manilow2019cutting,
  title={Cutting Music Source Separation Some {Slakh}: A Dataset to Study the Impact of Training Data Quality and Quantity},
  author={Manilow, Ethan and Wichern, Gordon and Seetharaman, Prem and Le Roux, Jonathan},
  booktitle={Proc. IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  year={2019},
  organization={IEEE}
}
"""

INDEXES = {
    "default": "2100-redux",
    "test": "baby",
    "2100-redux": core.Index(filename="slakh_index_2100-redux.json", partial_download=['2100-redux']),
    "baby": core.Index(filename="slakh_index_baby.json", partial_download=['baby'])
}

REMOTES = {
    '2100-redux': download_utils.RemoteFileMetadata(
        filename='slakh2100_flac_redux.tar.gz',
        url='https://zenodo.org/record/4599666/files/slakh2100_flac_redux.tar.gz?download=1',
        checksum='f4b71b6c45ac9b506f59788456b3f0c4',
    ),
    'baby': download_utils.RemoteFileMetadata(
        filename='babyslakh_16k.tar.gz',
        url='https://zenodo.org/record/4603870/files/babyslakh_16k.tar.gz?download=1',
        checksum='311096dc2bde7d61c97e930edbfc7f78',
    )
}

LICENSE_INFO = """
Creative Commons Attribution 4.0 International
"""


class Track(core.Track):
    """slakh track class
    # -- YOU CAN AUTOMATICALLY GENERATE THIS DOCSTRING BY CALLING THE SCRIPT:
    # -- `scripts/print_track_docstring.py my_dataset`
    # -- note that you'll first need to have a test track (see "Adding tests to your dataset" below)

    Args:
        track_id (str): track id of the track

    Attributes:
        track_id (str): track id
        # -- Add any of the dataset specific attributes here

    Cached Properties:
        annotation (EventData): a description of this annotation

    """
    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        
        super().__init__(
            track_id,
            data_home,
            dataset_name=dataset_name,
            index=index,
            metadata=metadata,
        )
        
        self.mtrack_id = self.track_id.split("-")[0]
        self.audio_path = self.get_path("audio")
        self.midi_path = self.get_path("midi")
        self.metadata_path = self.get_path("metadata")

    @core.cached_property
    def _track_metadata(self):
        with open(self.metadata_path, 'r') as fhandle:
            metadata = yaml.safe_load(fhandle)
        return metadata["stems"][self.track_id.split("-")[1]]

    @property
    def instrument(self):
        return self._track_metadata["inst_class"]
    
    @property
    def integrated_loudness(self):
        return self._track_metadata["integrated_loudness"]
    
    @property
    def is_drum(self):
        return self._track_metadata["is_drum"]

    @property
    def midi_program_name(self):
        return self._track_metadata["midi_program_name"]

    @property
    def plugin_name(self):
        return self._track_metadata["plugin_name"]

    @property
    def program_number(self):
        return self._track_metadata["program_num"]

    @core.cached_property
    def midi(self) -> Optional[pretty_midi.PrettyMidi]:
        """output type: description of output"""
        return load_midi(self.midi_path)

    @core.cached_property
    def notes(self) -> Optional[annotations.NoteData]:
        """output type: description of output"""
        return load_notes(self.midi_path)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """(np.ndarray, float): DESCRIPTION audio signal, sample rate"""
        return load_audio(self.audio_path)

    # -- we use the to_jams function to convert all the annotations in the JAMS format.
    # -- The converter takes as input all the annotations in the proper format (e.g. beats
    # -- will be fed as beat_data=[(self.beats, None)], see jams_utils), and returns a jams
    # -- object with the annotations.
    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            annotation_data=[(self.annotation, None)],
            metadata=self._metadata,
        )
        # -- see the documentation for `jams_utils.jams_converter for all fields


# -- if the dataset contains multitracks, you can define a MultiTrack similar to a Track
# -- you can delete the block of code below if the dataset has no multitracks
class MultiTrack(core.MultiTrack):
    """slakh multitrack class

    Args:
        mtrack_id (str): multitrack id
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/slakh`

    Attributes:
        mtrack_id (str): track id
        tracks (dict): {track_id: Track}
        track_audio_attribute (str): the name of the attribute of Track which
            returns the audio to be mixed
        # -- Add any of the dataset specific attributes here

    Cached Properties:
        annotation (EventData): a description of this annotation

    """
    def __init__(self, mtrack_id, data_home):
        self.mtrack_id = mtrack_id
        self._data_home = data_home
        # these three attributes below must have exactly these names
        self.track_ids = [...] # define which track_ids should be part of the multitrack
        self.tracks = {t: Track(t, self._data_home) for t in self.track_ids}
        self.track_audio_property = "audio" # the property of Track which returns the relevant audio file for mixing

        # -- optionally add any multitrack specific attributes here
        self.mix_path = ...  # this can be called whatever makes sense for the datasets
        self.annotation_path = ...

    # -- multitracks can optionally have mix-level cached properties and properties
    @core.cached_property
    def annotation(self) -> Optional[annotations.EventData]:
        """output type: description of output"""
        return load_annotation(self.annotation_path)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """(np.ndarray, float): DESCRIPTION audio signal, sample rate"""
        return load_audio(self.audio_path)

    # -- multitrack classes are themselves Tracks, and also need a to_jams method
    # -- for any mixture-level annotations
    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.mix_path,
            annotation_data=[(self.annotation, None)],
            ...
        )
        # -- see the documentation for `jams_utils.jams_converter for all fields


# -- this decorator allows this function to take a string or an open bytes file as input
# -- and in either case converts it to an open file handle.
# -- It also checks if the file exists
# -- and, if None is passed, None will be returned 
@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a slakh audio file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        * np.ndarray - the audio signal
        * float - The sample rate of the audio file

    """
    # -- for slakh, the code below. This should be dataset specific!
    # -- By default we load to mono
    # -- change this if it doesn't make sense for your dataset.
    return librosa.load(audio_path, sr=None, mono=True)


# -- Write any necessary loader functions for loading the dataset's data

# -- this decorator allows this function to take a string or an open file as input
# -- and in either case converts it to an open file handle.
# -- It also checks if the file exists
# -- and, if None is passed, None will be returned 
@io.coerce_to_string_io
def load_annotation(fhandle: TextIO) -> Optional[annotations.EventData]:

    # -- because of the decorator, the file is already open
    reader = csv.reader(fhandle, delimiter=' ')
    intervals = []
    annotation = []
    for line in reader:
        intervals.append([float(line[0]), float(line[1])])
        annotation.append(line[2])

    annotation_data = annotations.EventData(
        np.array(intervals), np.array(annotation)
    )
    return annotation_data

# -- use this decorator so the docs are complete
@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The slakh dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name=NAME,
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    # -- Copy any loaders you wrote that should be part of the Dataset class
    # -- use this core.copy_docs decorator to copy the docs from the original
    # -- load_ function
    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_annotation)
    def load_annotation(self, *args, **kwargs):
        return load_annotation(*args, **kwargs)

    # -- if your dataset has a top-level metadata file, write a loader for it here
    # -- you do not have to include this function if there is no metadata 
    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, 'slakh_metadta.csv')

        # load metadata however makes sense for your dataset
        metadata_path = os.path.join(data_home, 'slakh_metadata.json')
        with open(metadata_path, 'r') as fhandle:
            metadata = json.load(fhandle)

        return metadata

    # -- if your dataset needs to overwrite the default download logic, do it here.
    # -- this function is usually not necessary unless you need very custom download logic
    def download(
        self, partial_download=None, force_overwrite=False, cleanup=False
    ):
        """Download the dataset

        Args:
            partial_download (list or None):
                A list of keys of remotes to partially download.
                If None, all data is downloaded
            force_overwrite (bool):
                If True, existing files are overwritten by the downloaded files. 
            cleanup (bool):
                Whether to delete any zip/tar files after extracting.

        Raises:
            ValueError: if invalid keys are passed to partial_download
            IOError: if a downloaded file's checksum is different from expected

        """
        # see download_utils.downloader for basic usage - if you only need to call downloader
        # once, you do not need this function at all.
        # only write a custom function if you need it!

