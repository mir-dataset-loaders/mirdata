# -*- coding: utf-8 -*-
"""Utility functions for mirdata"""

import hashlib
import os
import tqdm


def md5(file_path):
    """Get md5 hash of a file.

    Args:
        file_path (str): File path

    Returns:
        md5_hash (str): md5 hash of data in file_path

    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as fhandle:
        for chunk in iter(lambda: fhandle.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def log_message(message, verbose=True):
    """Helper function to log message

    Args:
        message (str): message to log
        verbose (bool): if false, the message is not logged
    """
    if verbose:
        print(message)


def validate(file_id, local_path, checksum, missing_files, invalid_checksums):

    # validate that the file exists on disk
    if not os.path.exists(local_path):
        if file_id not in missing_files.keys():
            missing_files[file_id] = []
        missing_files[file_id].append(local_path)
    # validate that the checksum matches
    elif md5(local_path) != checksum:
        if file_id not in invalid_checksums.keys():
            invalid_checksums[file_id] = []
        invalid_checksums[file_id].append(local_path)


def check_files(file_dict, data_home, verbose):
    missing = {}
    invalid = {}
    for file_id, file in tqdm.tqdm(file_dict.items(), disable=not verbose):
        # multitrack case
        if file_id is "tracks":
            continue
        # tracks
        else:
            for tracks in file.keys():
                filepath = file[tracks][0]
                checksum = file[tracks][1]
                if filepath is not None:
                    local_path = os.path.join(data_home, filepath)
                    validate(file_id, local_path, checksum, missing, invalid)
    return missing, invalid


def check_metadata(file_dict, data_home, verbose):
    missing = {}
    invalid = {}
    for file_id, file in tqdm.tqdm(file_dict.items(), disable=not verbose):
        filepath = file[0]
        checksum = file[1]
        if filepath is not None:
            local_path = os.path.join(data_home, filepath)
            validate(file_id, local_path, checksum, missing, invalid)
    return missing, invalid


def check_index(dataset_index, data_home, verbose=True):
    """check index to find out missing files and files with invalid checksum

    Args:
        dataset_index (list): dataset indices
        data_home (str): Local home path that the dataset is being stored
        verbose (bool): if true, prints validation status while running

    Returns:
        missing_files (list): List of file paths that are in the dataset index
            but missing locally
        invalid_checksums (list): List of file paths that file exists in the dataset
            index but has a different checksum compare to the reference checksum

    """
    missing_files = {}
    invalid_checksums = {}

    # check index
    if "metadata" in dataset_index and dataset_index["metadata"] is not None:
        missing_metadata, invalid_metadata = check_metadata(
            dataset_index["metadata"], data_home, verbose,
        )
        missing_files["metadata"] = missing_metadata
        invalid_checksums["metadata"] = invalid_metadata

    if "tracks" in dataset_index and dataset_index["tracks"] is not None:
        missing_tracks, invalid_tracks = check_files(
            dataset_index["tracks"], data_home, verbose,
        )
        missing_files["tracks"] = missing_tracks
        invalid_checksums["tracks"] = invalid_tracks

    if "multitracks" in dataset_index and dataset_index["multitracks"] is not None:
        missing_multitracks, invalid_multitracks = check_files(
            dataset_index["multitracks"], data_home, verbose,
        )
        missing_files["multitracks"] = missing_multitracks
        invalid_checksums["multitracks"] = invalid_multitracks

    return missing_files, invalid_checksums


def validator(dataset_index, data_home, verbose=True):
    """Checks the existence and validity of files stored locally with
    respect to the paths and file checksums stored in the reference index.
    Logs invalid checksums and missing files.

    Args:
        dataset_index (list): dataset indices
        data_home (str): Local home path that the dataset is being stored
        verbose (bool): if True (default), prints missing and invalid files
            to stdout. Otherwise, this function is equivalent to check_index.

    Returns:
        missing_files (list): List of file paths that are in the dataset index
            but missing locally.
        invalid_checksums (list): List of file paths that file exists in the
            dataset index but has a different checksum compare to the reference
            checksum.
    """
    missing_files, invalid_checksums = check_index(dataset_index, data_home, verbose)

    # print path of any missing files
    has_any_missing_file = False
    for file_id in missing_files:
        if len(missing_files[file_id]) > 0:
            log_message("Files missing for {}:".format(file_id), verbose)
            for fpath in missing_files[file_id]:
                log_message(fpath, verbose)
            log_message("-" * 20, verbose)
            has_any_missing_file = True

    # print path of any invalid checksums
    has_any_invalid_checksum = False
    for file_id in invalid_checksums:
        if len(invalid_checksums[file_id]) > 0:
            log_message("Invalid checksums for {}:".format(file_id), verbose)
            for fpath in invalid_checksums[file_id]:
                log_message(fpath, verbose)
            log_message("-" * 20, verbose)
            has_any_invalid_checksum = True

    if not (has_any_missing_file or has_any_invalid_checksum):
        log_message(
            "Success: the dataset is complete and all files are valid.", verbose
        )
        log_message("-" * 20, verbose)

    return missing_files, invalid_checksums

