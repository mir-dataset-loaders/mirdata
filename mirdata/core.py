# -*- coding: utf-8 -*-
"""core mirdata classes
"""
import importlib
import os
import random
import types
import numpy as np

import mirdata
from mirdata import download_utils
from mirdata import utils

MAX_STR_LEN = 100
DATASETS = mirdata.DATASETS


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
        module = importlib.import_module("mirdata.datasets.{}".format(dataset))
        self.name = dataset
        self.bibtex = getattr(module, "BIBTEX", None)
        self._remotes = getattr(module, "REMOTES", None)
        self._index = module.DATA.index
        self._download_info = getattr(module, "DOWNLOAD_INFO", None)
        self._track_object = getattr(module, "Track", None)
        self._download_fn = getattr(module, "_download", download_utils.downloader)
        self._readme_str = module.__doc__

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
        repr_string += "-" * MAX_STR_LEN
        repr_string += "\n"
        repr_string += (
            "Call the .readme method for complete documentation of this dataset.\n"
        )
        repr_string += "Call the .cite method for bibtex citations.\n"
        repr_string += "-" * MAX_STR_LEN
        repr_string += "\n"
        if self._track_object is not None:
            repr_string += self.track.__doc__
            repr_string += "-" * MAX_STR_LEN
            repr_string += "\n"

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

    def readme(self):
        """Print the dataset's readme.
        """
        print(self._readme_str)

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


class MultiTrack(Track):
    """MultiTrack class.

    A multitrack class is a collection of track objects and their associated audio
    that can be mixed together.
    A multitrack is iteslf a Track, and can have its own associated audio (such as
    a mastered mix), its own metadata and its own annotations.

    """

    def _check_mixable(self):
        if not hasattr(self, "tracks") or not hasattr(self, "track_audio_property"):
            raise NotImplementedError(
                "This MultiTrack has no tracks/track_audio_property. Cannot perform mixing"
            )

    def get_target(self, track_keys, weights=None, average=True, enforce_length=True):
        """Get target which is a linear mixture of tracks

        Args:
            track_keys (list): list of track keys to mix together
            weights (list or None): list of positive scalars to be used in the average
            average (bool): if True, computes a weighted average of the tracks
                if False, computes a weighted sum of the tracks
            enforce_length (bool): If True, raises ValueError if the tracks are 
                not the same length. If False, pads audio with zeros to match the length
                of the longest track
        
        Returns:
            target (np.ndarray): target audio with shape (n_channels, n_samples)

        Raises:
            ValueError: 
                if sample rates of the tracks are not equal
                if enforce_length=True and lengths are not equal

        """
        self._check_mixable()
        signals = []
        lengths = []
        sample_rates = []
        for k in track_keys:
            audio, sample_rate = getattr(self.tracks[k], self.track_audio_property)
            # ensure all signals are shape (n_channels, n_samples)
            if len(audio.shape) == 1:
                audio = audio[np.newaxis, :]
            signals.append(audio)
            lengths.append(audio.shape[1])
            sample_rates.append(sample_rate)

        if len(set(sample_rates)) > 1:
            raise ValueError(
                "Sample rates for tracks {} are not equal: {}".format(
                    track_keys, sample_rates
                )
            )

        max_length = np.max(lengths)
        if any([l != max_length for l in lengths]):
            if enforce_length:
                raise ValueError(
                    "Track's {} audio are not the same length {}. Use enforce_length=False to pad with zeros.".format(
                        track_keys, lengths
                    )
                )
            else:
                # pad signals to the max length
                signals = [
                    np.pad(signal, ((0, 0), (0, max_length - signal.shape[1])))
                    for signal in signals
                ]

        if weights is None:
            weights = np.ones((len(track_keys),))

        target = np.average(signals, axis=0, weights=weights)
        if not average:
            target *= np.sum(weights)

        return target

    def get_random_target(self, n_tracks=None, min_weight=0.3, max_weight=1.0):
        """Get a random target by combining a random selection of tracks with random weights

        Args:
            n_tracks (int or None): number of tracks to randomly mix. If None, uses all tracks
            min_weight (float): minimum possible weight when mixing
            max_weight (float): maximum possible weight when mixing
        
        Returns:
            target (np.ndarray): mixture audio with shape (n_samples, n_channels)
            tracks (list): list of keys of included tracks
            weights (list): list of weights used to mix tracks
        """
        self._check_mixable()
        tracks = list(self.tracks.keys())
        if n_tracks is not None and n_tracks < len(tracks):
            tracks = np.random.choice(tracks, n_tracks, replace=False)

        weights = np.random.uniform(low=min_weight, high=max_weight, size=len(tracks))
        target = self.get_target(tracks, weights=weights)
        return target, tracks, weights

    def get_mix(self):
        """Create a linear mixture given a subset of tracks.

        Args:
            track_keys (list): list of track keys to mix together
        
        Returns:
            target (np.ndarray): mixture audio with shape (n_samples, n_channels)
        """
        self._check_mixable()
        return self.get_target(list(self.tracks.keys()))
