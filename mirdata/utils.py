# -*- coding: utf-8 -*-
"""Utility functions for mirdata

Attributes:
    MIR_DATASETS_DIR (str): home folder for MIR datasets

    NoteData (namedtuple): `intervals`, `notes`, `confidence`

    F0Data (namedtuple): `times`, `frequencies`, `confidence`

    LyricData (namedtuple): `start_times`, `end_times`, `lyrics`, `pronounciations`

    SectionData (namedtuple): `start_times`, `end_times`, `sections`

    BeatData (namedtuple): `beat_times`, `beat_positions`

    ChordData (namedtuple): `start_times`, `end_times`, `chords`

    KeyData (namedtuple): `start_times`, '`end_times`, `keys`

    EventData (namedtuple): `start_times`, `end_times`, `event`

    TempoData (namedtuple): `time`, `duration`, `value`, `confidence`

"""


from collections import namedtuple
import hashlib
import os
import json


MIR_DATASETS_DIR = os.path.join(os.getenv('HOME', '/tmp'), 'mir_datasets')


def md5(file_path):
    """Get md5 hash of a file.

    Args:
        file_path (str): File path

    Returns:
        md5_hash (str): md5 hash of data in file_path

    """
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as fhandle:
        for chunk in iter(lambda: fhandle.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def none_path_join(partial_path_list):
    """Join a list of partial paths. If any part of the path is None,
    returns None.

    Args:
        partial_path_list (list): List of partial paths

    Returns:
        path or None (str or None): joined path string or None

    """
    if None in partial_path_list:
        return None
    else:
        return os.path.join(*partial_path_list)


def log_message(message, silence=False):
    """Helper function to log message

    Args:
        message (str): message to log
        silence (bool): if true, the message is not logged
    """
    if not silence:
        print(message)


def check_index(dataset_index, data_home):
    """check index to find out missing files and files with invalid checksum

    Args:
        dataset_index (list): dataset indices
        data_home (str): Local home path that the dataset is being stored

    Returns:
        missing_files (list): List of file paths that are in the dataset index
            but missing locally
        invalid_checksums (list): List of file paths that file exists in the dataset
            index but has a different checksum compare to the reference checksum

    """
    missing_files = {}
    invalid_checksums = {}

    # loop over track ids
    for track_id, track in dataset_index.items():
        # loop over each data file for this track id
        for key in track.keys():
            filepath = track[key][0]
            checksum = track[key][1]
            if filepath is not None:
                local_path = os.path.join(data_home, filepath)
                # validate that the file exists on disk
                if not os.path.exists(local_path):
                    if track_id not in missing_files.keys():
                        missing_files[track_id] = []
                    missing_files[track_id].append(local_path)
                # validate that the checksum matches
                elif md5(local_path) != checksum:
                    if track_id not in invalid_checksums.keys():
                        invalid_checksums[track_id] = []
                    invalid_checksums[track_id].append(local_path)

    return missing_files, invalid_checksums


def validator(dataset_index, data_home, silence=False):
    """Checks the existence and validity of files stored locally with
    respect to the paths and file checksums stored in the reference index.
    Logs invalid checksums and missing files.

    Args:
        dataset_index (list): dataset indices
        data_home (str): Local home path that the dataset is being stored
        silence (bool): if False (default), prints missing and invalid files
        to stdout. Otherwise, this function is equivalent to check_index.

    Returns:
        missing_files (list): List of file paths that are in the dataset index
            but missing locally.
        invalid_checksums (list): List of file paths that file exists in the
            dataset index but has a different checksum compare to the reference
            checksum.
    """
    missing_files, invalid_checksums = check_index(dataset_index, data_home)

    # print path of any missing files
    has_any_missing_file = False
    for track_id in missing_files.keys():
        if len(missing_files[track_id]) > 0:
            log_message('Files missing for {}:'.format(track_id), silence)
            for fpath in missing_files[track_id]:
                log_message(fpath, silence)
            log_message('-' * 20, silence)
            has_any_missing_file = True

    # print path of any invalid checksums
    has_any_invalid_checksum = False
    for track_id in invalid_checksums.keys():
        if len(invalid_checksums[track_id]) > 0:
            log_message('Invalid checksums for {}:'.format(track_id), silence)
            for fpath in invalid_checksums[track_id]:
                log_message(fpath, silence)
            log_message('-' * 20, silence)
            has_any_invalid_checksum = True

    if not (has_any_missing_file or has_any_invalid_checksum):
        log_message(
            'Success: the dataset is complete and all files are valid.', silence
        )
        log_message('-' * 20, silence)

    return missing_files, invalid_checksums


NoteData = namedtuple('NoteData', ['intervals', 'notes', 'confidence'])

F0Data = namedtuple('F0Data', ['times', 'frequencies', 'confidence'])

MultipitchData = namedtuple(
    'MultipitchData', ['times', 'frequency_list', 'confidence_list']
)

LyricData = namedtuple(
    'LyricData', ['start_times', 'end_times', 'lyrics', 'pronunciations']
)

SectionData = namedtuple('SectionData', ['intervals', 'labels'])

BeatData = namedtuple('BeatData', ['beat_times', 'beat_positions'])

ChordData = namedtuple('ChordData', ['intervals', 'labels'])

KeyData = namedtuple('KeyData', ['start_times', 'end_times', 'keys'])

TempoData = namedtuple('TempoData', ['time', 'duration', 'value', 'confidence'])

EventData = namedtuple('EventData', ['start_times', 'end_times', 'event'])


def get_default_dataset_path(dataset_name):
    """Get the default path for a dataset given it's name

    Args:
        dataset_name (str or None)
            The name of the dataset folder, e.g. 'Orchset'

    Returns:
        save_path (str): Local path to the dataset
    """
    return os.path.join(MIR_DATASETS_DIR, dataset_name)


def load_json_index(filename):
    CWD = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(CWD, 'indexes', filename)) as f:
        return json.load(f)


class cached_property(object):
    """A property that is only computed once per instance and then replaces
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


class LargeData(object):
    def __init__(self, index_file, metadata_load_fn=None):
        """Object which loads and caches large data the first time it's
        accessed.

        Parameters
        ----------
        index_file: str
            File name of checksum index file to be passed to `load_json_index`
        metadata_load_fn: function
            Function which returns a metadata dictionary.
            If None, assume the dataset has no metadata. When the
            `metadata` attribute is called, raises a NotImplementedError

        """
        self._metadata = None
        self.index_file = index_file
        self.metadata_load_fn = metadata_load_fn

    @cached_property
    def index(self):
        return load_json_index(self.index_file)

    def metadata(self, data_home):
        if self.metadata_load_fn is None:
            raise NotImplementedError

        if self._metadata is None or self._metadata['data_home'] != data_home:
            self._metadata = self.metadata_load_fn(data_home)
        return self._metadata
