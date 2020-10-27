# -*- coding: utf-8 -*-
"""core mirdata classes
"""
import importlib
import os
import random
import types

import mirdata
from mirdata import download_utils
from mirdata import utils

MAX_STR_LEN = 100
DATASETS = mirdata.__all__


class Dataset(object):
    """mirdata Dataset object

    Usage example:
    orchset = mirdata.Dataset('orchset')  # get the orchset dataset
    orchset.download()  # download orchset
    orchset.validate()  # validate orchset
    track = orchset.choice_track()  # load a random track
    print(track)  # see what data a track contains
    orchset.track_ids()  # load all track ids

    Attributes:
        name (str): the identifier of the dataset
        bibtex (str): dataset citation/s in bibtex format
        remotes (dict): data to be downloaded
        index (dict): dataset file index
        download_info (str): download instructions or caveats
        track (mirdata.core.Track): function that inputs a track_id
        readme (str): information about the dataset
        data_home (str): path where mirdata will look for the dataset

    """

    def __init__(self, dataset, data_home=None):
        """Inits a dataset by name and data location"""
        if dataset not in DATASETS:
            raise ValueError(
                "{} is not a valid dataset in mirdata. Valid datsets are:\n{}".format(
                    dataset, ",".join(DATASETS)
                )
            )
        module = importlib.import_module("mirdata.{}".format(dataset))
        self.name = dataset
        self.bibtex = getattr(module, "BIBTEX", None)
        self._remotes = getattr(module, "REMOTES", None)
        self._index = module.DATA.index
        self._download_info = getattr(module, "DOWNLOAD_INFO", None)
        self._track_object = getattr(module, "Track", None)
        self._download_fn = getattr(module, "_download", download_utils.downloader)
        self.readme = module.__doc__

        if data_home is None:
            self.data_home = self.default_path
        else:
            self.data_home = data_home

        # this is a hack to be able to have dataset-specific docstrings
        self.track = lambda track_id: self._track(track_id)
        self.track.__doc__ = self._track_object.__doc__  # set the docstring

        # inherit any public load functions from the module
        for method_name in dir(module):
            if method_name.startswith("load_"):
                method = getattr(module, method_name)
                setattr(self, method_name, method)
                # getattr(self, method_name).__doc__ = method.__doc__

    def __repr__(self):
        repr_string = "The {} dataset\n".format(self.name)
        repr_string += (
            "Call the .readme method for complete documentation of this dataset."
        )
        repr_string += "Call the .cite method for bibtex citations.\n"
        repr_string += "=" * MAX_STR_LEN
        repr_string += "\n"
        if self._track_object is not None:
            repr_string += "Tracks:\n"
            repr_string += self.track.__repr__

        return repr_string

    @property
    def default_path(self):
        """Get the default path for the dataset

        Returns:
            default_path (str): Local path to the dataset
        """
        mir_datasets_dir = os.path.join(os.getenv("HOME", "/tmp"), "mir_datasets")
        return os.path.join(mir_datasets_dir, self.name)

    def _track(self, track_id):
        """Load a track by track_id.
        Hidden helper function that gets called as a lambda.

        Args:
            track_id (str): track id of the track
        
        Returns:
            track (dataset.Track): an instance of this dataset's Track object
        """
        if self._track_object is None:
            raise NotImplementedError
        else:
            return self._track_object(track_id, self.data_home)

    def load_tracks(self):
        """Load all tracks in the dataset

        Returns:
            (dict): {`track_id`: track data}
        
        Raises:
            NotImplementedError: If the dataset does not support Track objects
        """
        return {track_id: self.track(track_id) for track_id in self.track_ids}

    def choice_track(self):
        """Choose a random track

        Returns:
            track (dataset.Track): a random Track object
        """
        return self.track(random.choice(self.track_ids))

    def cite(self):
        """Print the reference"""
        print("========== BibTeX ==========")
        print(self.bibtex)

    def download(self, partial_download=None, force_overwrite=False, cleanup=True):
        """Download data to `save_dir` and optionally print a message.

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
        self._download_fn(
            self.data_home,
            remotes=self._remotes,
            partial_download=partial_download,
            info_message=self._download_info,
            force_overwrite=force_overwrite,
            cleanup=cleanup,
        )

    @utils.cached_property
    def track_ids(self):
        """Return track ids

        Returns:
            (list): A list of track ids
        """
        return list(self._index.keys())

    def validate(self, verbose=True):
        """Validate if the stored dataset is a valid version

        Args:
            verbose (bool): If False, don't print output

        Returns:
            missing_files (list): List of file paths that are in the dataset index
                but missing locally
            invalid_checksums (list): List of file paths that file exists in the dataset
                index but has a different checksum compare to the reference checksum

        """
        missing_files, invalid_checksums = utils.validator(
            self._index, self.data_home, verbose=verbose
        )
        return missing_files, invalid_checksums


class Track(object):
    def __repr__(self):
        properties = [v for v in dir(self.__class__) if not v.startswith("_")]
        attributes = [
            v for v in dir(self) if not v.startswith("_") and v not in properties
        ]

        repr_str = "Track(\n"

        for attr in attributes:
            val = getattr(self, attr)
            if isinstance(val, str):
                if len(val) > MAX_STR_LEN:
                    val = "...{}".format(val[-MAX_STR_LEN:])
                val = '"{}"'.format(val)
            repr_str += "  {}={},\n".format(attr, val)

        for prop in properties:
            val = getattr(self.__class__, prop)
            if isinstance(val, types.FunctionType):
                continue

            if val.__doc__ is None:
                raise ValueError("{} has no documentation".format(prop))

            val_type_str = val.__doc__.split(":")[0]
            repr_str += "  {}: {},\n".format(prop, val_type_str)

        repr_str += ")"
        return repr_str

    def to_jams(self):
        raise NotImplementedError

