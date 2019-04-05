# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import hashlib
import os
import json
import tarfile
try:
    from urllib.request import urlopen  # py3
except ImportError:
    from urllib2 import urlopen  # py2
try:
    from urllib.request import urlretrieve  # py3
except ImportError:
    from urllib import urlretrieve  # py2
import zipfile
from tqdm import tqdm

MIR_DATASETS_DIR = os.path.join(os.environ["HOME"], "mir_datasets")


def md5(file_path):
    """Get md5 hash of a file.

    Parameters
    ----------
    file_path: str
        File path.

    Returns
    -------
    md5_hash: str
        md5 hash of data in file_path
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as fhandle:
        for chunk in iter(lambda: fhandle.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def log_message(message, silence=False):
    if not silence:
        print(message)


def validator(dataset_index, data_home, silence=False):
    missing_files = {}
    invalid_checksums = {}

    # loop over track ids
    for track_id, track in dataset_index.items():
        # loop over each data file for this track id
        for key in track.keys():
            filepath = track[key][0]
            checksum = track[key][1]
            local_path = get_local_path(data_home, filepath)
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

    # print path of any missing files
    for track_id in missing_files.keys():
        if len(missing_files[track_id]) > 0:
            log_message("Files missing for {}:".format(track_id), silence)
            for fpath in missing_files[track_id]:
                log_message(fpath, silence)
            log_message("-" * 20, silence)

    # print path of any invalid checksums
    for track_id in invalid_checksums.keys():
        if len(invalid_checksums[track_id]) > 0:
            log_message("Invalid checksums for {}:".format(track_id), silence)
            for fpath in invalid_checksums[track_id]:
                log_message(fpath, silence)
            log_message("-" * 20, silence)

    return missing_files, invalid_checksums


F0Data = namedtuple(
    'F0Data',
    ['times', 'frequencies', 'confidence']
)

LyricsData = namedtuple(
    'LyricsData',
    ['start_time', 'end_time', 'lyric', 'pronounciation']
)


def get_local_path(data_home, rel_path):
    if data_home is None:
        return os.path.join(MIR_DATASETS_DIR, rel_path)
    else:
        return os.path.join(data_home, rel_path)


def get_save_path(data_home):
    """Get path to save a file given value of `data_home`, and create it if it
    does not exist.

    Parameters
    ----------
    data_home: str or None
        If string, `save_path` is set to data_home.
        If None, `save_path` is set to the default MIR_DATASETS_DIR value.

    Returns
    ------_
    save_path: str
        Path to save data.
    """
    if data_home is None:
        save_path = MIR_DATASETS_DIR
    else:
        save_path = data_home

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    return save_path


RemoteFileMetadata = namedtuple('RemoteFileMetadata',
                                ['filename', 'url', 'checksum'])


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_from_remote(remote, data_home=None, clobber=False):
    """Download a remote dataset into path
    Fetch a dataset pointed by remote's url, save into path using remote's
    filename and ensure its integrity based on the MD5 Checksum of the
    downloaded file.

    Adapted from scikit-learn's sklearn.datasets.base._fetch_remote.

    Parameters
    -----------
    remote: RemoteFileMetadata
        Named tuple containing remote dataset meta information: url, filename
        and checksum
    data_home: string
        Directory to save the file to.
    clobber: bool
        If True, overwrite existing file with the downloaded file.
        If False, does not overwrite, but checks that checksum is consistent.

    Returns
    -------
    file_path: string
        Full path of the created file.
    """
    download_path = (
        os.path.join(MIR_DATASETS_DIR, remote.filename) if data_home is None
        else os.path.join(data_home, remote.filename)
    )

    if not os.path.exists(download_path) or clobber:
        # If file doesn't exist or we want to overwrite, download it
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1,
                                 desc=remote.url.split('/')[-1]) as t:
            urlretrieve(
                remote.url, filename=download_path, reporthook=t.update_to)

    checksum = md5(download_path)
    if remote.checksum != checksum:
        raise IOError("{} has an MD5 checksum ({}) "
                      "differing from expected ({}), "
                      "file may be corrupted.".format(download_path, checksum,
                                                      remote.checksum))
    return download_path


def unzip(zip_path, save_dir, cleanup=False):
    """Unzip a zip file to a specified save location.

    Parameters
    ----------
    zip_path: str
        Path to zip file
    save_dir: str
        Path to save unzipped data
    cleanup: bool, default=False
        If True, remove zipfile after unzipping.
    """
    zfile = zipfile.ZipFile(zip_path, 'r')
    zfile.extractall(save_dir)
    zfile.close()
    if cleanup:
        os.remove(zip_path)


def untar(tar_path, save_dir, cleanup=False):
    """Untar a tar file to a specified save location.

    Parameters
    ----------
    tar_path: str
        Path to tar file
    save_dir: str
        Path to save untarred data
    cleanup: bool, default=False
        If True, remove tarfile after untarring.
    """
    tfile = tarfile.TarFile(tar_path, 'r')
    tfile.extractall(save_dir)
    tfile.close()
    if cleanup:
        os.remove(tar_path)


def load_json_index(filename):
    CWD = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(CWD, 'indexes', filename)) as f:
        return json.load(f)
