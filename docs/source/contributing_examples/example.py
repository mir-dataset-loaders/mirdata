# -*- coding: utf-8 -*-
"""Example Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    Please include the following information at the top level docstring for the dataset's module `dataset.py`:

    1. Describe annotations included in the dataset
    2. Indicate the size of the datasets (e.g. number files and duration, hours)
    3. Mention the origin of the dataset (e.g. creator, institution)
    4. Describe the type of music included in the dataset
    5. Indicate any relevant papers related to the dataset
    6. Include a description about how the data can be accessed and the license it uses (if applicable)

"""
import csv
import logging
import json
import os

import librosa
import numpy as np
# -- import whatever you need here and remove
# -- example imports you won't use

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core, annotations

# -- Add any relevant citations here
BIBTEX = """
@article{article-minimal,
  author = "L[eslie] B. Lamport",
  title = "The Gnats and Gnus Document Preparation System",
  journal = "G-Animal's Journal",
  year = "1986"
}
"""

# -- REMOTES is a dictionary containing all files that need to be downloaded.
# -- The keys should be descriptive (e.g. 'annotations', 'audio').
# -- When having data that can be partially downloaded, remember to set up
# -- correctly destination_dir to download the files following the correct structure.
REMOTES = {
    'remote_data': download_utils.RemoteFileMetadata(
        filename='a_zip_file.zip',
        url='http://website/hosting/the/zipfile.zip',
        checksum='00000000000000000000000000000000',  # -- the md5 checksum
        destination_dir='path/to/unzip' # -- relative path for where to unzip the data, or None
    ),
}

# -- Include any information that should be printed when downloading
# -- remove this variable if you don't need to print anything during download
DOWNLOAD_INFO = """
Include any information you want to be printed when dataset.download() is called.
These can be instructions for how to download the dataset (e.g. request access on zenodo),
caveats about the download, etc
"""

# -- Include the dataset's license information
LICENSE_INFO = """
The dataset's license information goes here.
"""

DATA = core.LargeData('example_index.json')


class Track(core.Track):
    """Example track class
    # -- YOU CAN AUTOMATICALLY GENERATE THIS DOCSTRING BY CALLING THE SCRIPT:
    # -- `scripts/print_track_docstring.py my_dataset`
    # -- note that you'll first need to have a test track (see "Adding tests to your dataset" below)

    Args:
        track_id (str): track id of the track

    Attributes:
        track_id (str): track id
        # -- Add any of the dataset specific attributes here

    """
    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        
        # -- this sets the following attributes:
        # -- * track_id
        # -- * _dataset_name
        # -- * _data_home
        # -- * _track_paths
        # -- * _track_metadata
        super().__init__(
            track_id,
            data_home,
            dataset_name=dataset_name,
            index=index,
            metadata=metadata,
        )
        
        # -- add any dataset specific attributes here
        self.audio_path = os.path.join(
            self._data_home, self._track_paths['audio'][0])
        self.annotation_path = os.path.join(
            self._data_home, self._track_paths['annotation'][0])

    # -- `annotation` will behave like an attribute, but it will only be loaded
    # -- and saved when someone accesses it. Useful when loading slightly
    # -- bigger files or for bigger datasets. By default, we make any time
    # -- series data loaded from a file a cached property
    @core.cached_property
    def annotation(self):
        """output type: description of output"""
        return load_annotation(self.annotation_path)

    # -- `audio` will behave like an attribute, but it will only be loaded
    # -- when someone accesses it and it won't be stored. By default, we make
    # -- any memory heavy information (like audio) properties
    @property
    def audio(self):
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
    """Example multitrack class

    Args:
        mtrack_id (str): multitrack id
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/Example`

    Attributes:
        mtrack_id (str): track id
        tracks (dict): {track_id: Track}
        track_audio_attribute (str): the name of the attribute of Track which
            returns the audio to be mixed
        # -- Add any of the dataset specific attributes here

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
    def annotation(self):
        """output type: description of output"""
        return load_annotation(self.annotation_path)

    @property
    def audio(self):
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


@io.coerce_to_bytes_io
def load_audio(fhandle):
    """Load a Example audio file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        * np.ndarray - the audio signal
        * float - The sample rate of the audio file

    """
    # -- for example, the code below. This should be dataset specific!
    # -- By default we load to mono
    # -- change this if it doesn't make sense for your dataset.
    return librosa.load(audio_path, sr=None, mono=True)


# -- Write any necessary loader functions for loading the dataset's data
@io.coerce_to_string_io
def load_annotation(fhandle):

    # -- if there are some file paths for this annotation type in this dataset's
    # -- index that are None/null, uncomment the lines below.
    # if annotation_path is None:
    #     return None

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
    """The Example dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            index=DATA.index,
            name=NAME,
            track_class=Track,
            bibtex=BIBTEX,
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
        metadata_path = os.path.join(self.data_home, 'example_metadta.csv')

        # load metadata however makes sense for your dataset
        metadata_path = os.path.join(data_home, 'example_metadata.json')
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

