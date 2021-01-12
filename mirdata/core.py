# -*- coding: utf-8 -*-
"""Core mirdata classes
"""
import json
import os
import random
import types
import numpy as np

from mirdata import download_utils
from mirdata import validate

MAX_STR_LEN = 100
DOCS_URL = "https://mirdata.readthedocs.io/en/latest/source/mirdata.html"
DISCLAIMER = """
******************************************************************************************
DISCLAIMER: mirdata is a software package with its own license which is independent from
this dataset's license. We don not take responsibility for possible inaccuracies in the
license information provided in mirdata. It is the user's responsibility to be informed
and respect the dataset's license.
******************************************************************************************
"""

##### decorators ######


class cached_property(object):
    """Cached propery decorator

    A property that is only computed once per instance and then replaces
    itself with an ordinary attribute. Deleting the attribute resets the
    property.
    Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76

    """

    def __init__(self, func):
        self.__doc__ = getattr(func, "__doc__")
        self.func = func

    def __get__(self, obj, cls):
        # type: (Any, type) -> Any
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


def docstring_inherit(parent):
    """Decorator function to inherit docstrings from the parent class.

    Adds documented Attributes from the parent to the child docs.

    """

    def inherit(obj):
        spaces = "    "
        if not str(obj.__doc__).__contains__("Attributes:"):
            obj.__doc__ += "\n" + spaces + "Attributes:\n"
        obj.__doc__ = str(obj.__doc__).rstrip() + "\n"
        for attribute in parent.__doc__.split("Attributes:\n")[-1].lstrip().split("\n"):
            obj.__doc__ += spaces * 2 + str(attribute).lstrip().rstrip() + "\n"

        return obj

    return inherit


def copy_docs(original):
    """
    Decorator function to copy docs from one function to another
    """

    def wrapper(target):
        target.__doc__ = original.__doc__
        return target

    return wrapper


##### Core Classes #####


class Dataset(object):
    """mirdata Dataset object

    Attributes:
        data_home (str): path where mirdata will look for the dataset
        name (str): the identifier of the dataset
        bibtex (str or None): dataset citation/s in bibtex format
        remotes (dict or None): data to be downloaded
        readme (str): information about the dataset
        track (function): a function which inputs a track_id (str) and
            returns (mirdata.core.Track or None)

    """

    def __init__(
        self,
        data_home=None,
        index=None,
        name=None,
        track_object=None,
        bibtex=None,
        remotes=None,
        download_info=None,
        license_info=None,
    ):
        """Dataset init method

        Args:
            data_home (str or None): path where mirdata will look for the dataset
            index (dict or None): the dataset's file index
            name (str or None): the identifier of the dataset
            track_object (mirdata.core.Track or None): an uninstantiated Track object
            bibtex (str or None): dataset citation/s in bibtex format
            remotes (dict or None): data to be downloaded
            download_info (str or None): download instructions or caveats
            license_info (str or None): license of the dataset

        """
        self.name = name
        self.data_home = self.default_path if data_home is None else data_home
        self._index = index
        self._track_object = track_object
        self.bibtex = bibtex
        self.remotes = remotes
        self._download_info = download_info
        self._license_info = license_info
        self.readme = "{}#module-mirdata.datasets.{}".format(DOCS_URL, self.name)

        # this is a hack to be able to have dataset-specific docstrings
        self.track = lambda track_id: self._track(track_id)
        self.track.__doc__ = self._track_object.__doc__  # set the docstring

    def __repr__(self):
        repr_string = "The {} dataset\n".format(self.name)
        repr_string += "-" * MAX_STR_LEN
        repr_string += "\n\n\n"
        repr_string += "Call the .cite method for bibtex citations.\n"
        repr_string += "-" * MAX_STR_LEN
        repr_string += "\n\n\n"
        if self._track_object is not None:
            repr_string += self.track.__doc__
            repr_string += "-" * MAX_STR_LEN
            repr_string += "\n"

        return repr_string

    @property
    def default_path(self):
        """Get the default path for the dataset

        Returns:
            str: Local path to the dataset

        """
        mir_datasets_dir = os.path.join(os.getenv("HOME", "/tmp"), "mir_datasets")
        return os.path.join(mir_datasets_dir, self.name)

    def _track(self, track_id):
        """Load a track by track_id.

        Hidden helper function that gets called as a lambda.

        Args:
            track_id (str): track id of the track

        Returns:
           Track: an instance of this dataset's Track object

        """
        if self._track_object is None:
            raise NotImplementedError
        else:
            return self._track_object(track_id, self.data_home)

    def load_tracks(self):
        """Load all tracks in the dataset

        Returns:
            dict:
                {`track_id`: track data}

        Raises:
            NotImplementedError: If the dataset does not support Track objects

        """
        return {track_id: self.track(track_id) for track_id in self.track_ids}

    def choice_track(self):
        """Choose a random track

        Returns:
            Track: a Track object instantiated by a random track_id

        """
        return self.track(random.choice(self.track_ids))

    def cite(self):
        """
        Print the reference
        """
        print("========== BibTeX ==========")
        print(self.bibtex)

    def license(self):
        """
        Print the license
        """
        print("========== License ==========")
        print(self._license_info)
        print(DISCLAIMER)

    def download(self, partial_download=None, force_overwrite=False, cleanup=False):
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
        download_utils.downloader(
            self.data_home,
            remotes=self.remotes,
            partial_download=partial_download,
            info_message=self._download_info,
            force_overwrite=force_overwrite,
            cleanup=cleanup,
        )

    @cached_property
    def track_ids(self):
        """Return track ids

        Returns:
            list: A list of track ids

        """
        return list(self._index["tracks"].keys())

    def validate(self, verbose=True):
        """Validate if the stored dataset is a valid version

        Args:
            verbose (bool): If False, don't print output

        Returns:
            * list - files in the index but are missing locally
            * list - files which have an invalid checksum

        """
        missing_files, invalid_checksums = validate.validator(
            self._index, self.data_home, verbose=verbose
        )
        return missing_files, invalid_checksums


class Track(object):
    """Track base class

    See the docs for each dataset loader's Track class for details

    """

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
                doc = ""
            else:
                doc = val.__doc__

            val_type_str = doc.split(":")[0]
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
            np.ndarray: target audio with shape (n_channels, n_samples)

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
            * np.ndarray - mixture audio with shape (n_samples, n_channels)
            * list - list of keys of included tracks
            * list - list of weights used to mix tracks

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
            np.ndarray: mixture audio with shape (n_samples, n_channels)

        """
        self._check_mixable()
        return self.get_target(list(self.tracks.keys()))


def load_json_index(filename):
    working_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(working_dir, "datasets/indexes", filename)) as f:
        return json.load(f)


def none_path_join(partial_path_list):
    """Join a list of partial paths. If any part of the path is None,
    returns None.

    Args:
        partial_path_list (list): List of partial paths

    Returns:
        str or None: joined path string or None

    """
    if None in partial_path_list:
        return None
    else:
        return os.path.join(*partial_path_list)


class LargeData(object):
    def __init__(self, index_file, metadata_load_fn=None, remote_index=None):
        """Object which loads and caches large data the first time it's accessed.

        Args:
            index_file: str
                File name of checksum index file to be passed to `load_json_index`
            metadata_load_fn: function
                Function which returns a metadata dictionary.
                If None, assume the dataset has no metadata. When the
                `metadata` attribute is called, raises a NotImplementedError

        Cached Properties:
            index (dict): dataset index

        """
        self._metadata = None
        self.index_file = index_file
        self.metadata_load_fn = metadata_load_fn
        self.remote_index = remote_index

    @cached_property
    def index(self):
        if self.remote_index is not None:
            working_dir = os.path.dirname(os.path.realpath(__file__))
            path_index_file = os.path.join(
                working_dir, "datasets/indexes", self.index_file
            )
            if not os.path.isfile(path_index_file):
                path_indexes = os.path.join(working_dir, "datasets/indexes")
                download_utils.downloader(path_indexes, remotes=self.remote_index)
        return load_json_index(self.index_file)

    def metadata(self, data_home):
        """Dataset metadata

        Args:
            data_home (str): path where the dataset lives

        Raises:
            NotImplementedError: if self.metadata_load_fn is not set

        Returns:
            Object: data loaded by self.metadata_load_fn

        """
        if self.metadata_load_fn is None:
            raise NotImplementedError

        if self._metadata is None or self._metadata["data_home"] != data_home:
            self._metadata = self.metadata_load_fn(data_home)
        return self._metadata
