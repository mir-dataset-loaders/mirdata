"""Core mirdata classes
"""
import json
import os
import random
import types
from typing import Any

import numpy as np

from mirdata import download_utils
from mirdata import validate

MAX_STR_LEN = 100
DOCS_URL = "https://mirdata.readthedocs.io/en/stable/source/mirdata.html"
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

    def __get__(self, obj: Any, cls: type) -> Any:
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
    """mirdata Dataset class

    Attributes:
        data_home (str): path where mirdata will look for the dataset
        name (str): the identifier of the dataset
        bibtex (str or None): dataset citation/s in bibtex format
        remotes (dict or None): data to be downloaded
        readme (str): information about the dataset
        track (function): a function mapping a track_id to a mirdata.core.Track

    """

    def __init__(
        self,
        data_home=None,
        name=None,
        track_class=None,
        bibtex=None,
        remotes=None,
        download_info=None,
        license_info=None,
        custom_index_path=None,
    ):
        """Dataset init method

        Args:
            data_home (str or None): path where mirdata will look for the dataset
            name (str or None): the identifier of the dataset
            track_class (mirdata.core.Track or None): a Track class
            bibtex (str or None): dataset citation/s in bibtex format
            remotes (dict or None): data to be downloaded
            download_info (str or None): download instructions or caveats
            license_info (str or None): license of the dataset
            custom_index_path (str or None): overwrites the default index path for remote indexes

        """
        self.name = name
        self.data_home = self.default_path if data_home is None else data_home
        if custom_index_path:
            self.index_path = os.path.join(self.data_home, custom_index_path)
            self.remote_index = True
        else:
            self.index_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "datasets/indexes",
                "{}_index.json".format(self.name),
            )
            self.remote_index = False
        self._track_class = track_class
        self.bibtex = bibtex
        self.remotes = remotes
        self._download_info = download_info
        self._license_info = license_info
        self.readme = "{}#module-mirdata.datasets.{}".format(DOCS_URL, self.name)

        # this is a hack to be able to have dataset-specific docstrings
        self.track = lambda track_id: self._track(track_id)
        self.track.__doc__ = self._track_class.__doc__  # set the docstring

    def __repr__(self):
        repr_string = "The {} dataset\n".format(self.name)
        repr_string += "-" * MAX_STR_LEN
        repr_string += "\n\n\n"
        repr_string += "Call the .cite method for bibtex citations.\n"
        repr_string += "-" * MAX_STR_LEN
        repr_string += "\n\n\n"
        if self._track_class is not None:
            repr_string += self.track.__doc__
            repr_string += "-" * MAX_STR_LEN
            repr_string += "\n"

        return repr_string

    @cached_property
    def _index(self):
        if self.remote_index and not os.path.exists(self.index_path):
            raise FileNotFoundError(
                "This dataset's index is not available locally. You may need to first run .download()"
            )
        with open(self.index_path) as fhandle:
            index = json.load(fhandle)
        return index

    @cached_property
    def _metadata(self):
        return None

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
           Track: a Track object

        """
        if self._track_class is None:
            raise NotImplementedError
        else:
            return self._track_class(
                track_id,
                self.data_home,
                self.name,
                self._index,
                lambda: self._metadata,
            )

    def load_tracks(self):
        """Load all tracks in the dataset

        Returns:
            dict:
                {`track_id`: track data}

        Raises:
            NotImplementedError: If the dataset does not support Tracks

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

    def __init__(
        self,
        track_id,
        data_home,
        dataset_name,
        index,
        metadata=None,
    ):
        """Track init method. Sets boilerplate attributes, including:

        - ``track_id``
        - ``_dataset_name``
        - ``_data_home``
        - ``_track_paths``
        - ``_track_metadata``

        Args:
            track_id (str): track id
            data_home (str): path where mirdata will look for the dataset
            dataset_name (str): the identifier of the dataset
            index (dict): the dataset's file index
            metadata (dict or None): a dictionary of metadata or None

        """
        if track_id not in index["tracks"]:
            raise ValueError(
                "{} is not a valid track_id in {}".format(track_id, dataset_name)
            )

        self.track_id = track_id
        self._dataset_name = dataset_name

        self._data_home = data_home
        self._track_paths = index["tracks"][track_id]
        self._metadata = metadata

    @property
    def _track_metadata(self):
        metadata = self._metadata()
        if metadata and self.track_id in metadata:
            return metadata[self.track_id]
        elif metadata:
            return metadata
        return None

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
