# -*- coding: utf-8 -*-
"""functions for downloading from the web
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import os
import tarfile
import zipfile
import requests
from requests.exceptions import HTTPError
from tqdm import tqdm

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path  # python 2 backport

from mirdata.utils import md5


RemoteFileMetadata = namedtuple('RemoteFileMetadata', ['filename', 'url', 'checksum'])


def downloader(
    save_dir,
    zip_downloads=None,
    tar_downloads=None,
    file_downloads=None,
    info_message=None,
    force_overwrite=False,
):
    """Download data to `save_dir` and optionally print a message.

    Args:
        save_dir (str):
            The directory to download the data
        zip_downloads (list or None):
            A list of RemoteFileMetadata tuples of data in zip format.
            If None, there is no zip data to download
        tar_downloads (list or None):
            A list of RemoteFileMetadata tuples of data in tar format.
            If None, there is no tar data to download
        file_downloads (list or None):
            A list of  RemoteFileMetadata tuples of uncompressed data.
            If None, there is no uncompressed data to download.
        info_message (str or None):
            A string of info to print when this function is called.
            If None, no string is printed.
        force_overwrite (bool):
            If True, existing files are overwritten by the downloaded files.

    """
    Path(save_dir).mkdir(exist_ok=True)

    if zip_downloads is not None:
        for zip_download in zip_downloads:
            download_zip_file(zip_download, save_dir, force_overwrite)

    if tar_downloads is not None:
        for tar_download in tar_downloads:
            download_tar_file(tar_download, save_dir, force_overwrite)

    if file_downloads is not None:
        for file_download in file_downloads:
            download_from_remote(file_download, save_dir, force_overwrite)

    if info_message is not None:
        print(info_message)


class DownloadProgressBar(tqdm):
    """Wrap `tqdm` to show download progress"""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _download_large_file(url, download_path, callback=lambda: None):
    """download large file stably (todo: ??)"""
    response = requests.get(url)
    response.raise_for_status()
    with open(download_path, 'wb') as handle:
        for block in response.iter_content(4096):
            handle.write(block)
            callback()
    return download_path


def download_from_remote(remote, data_home, force_overwrite=False):
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
    force_overwrite: bool
        If True, overwrite existing file with the downloaded file.
        If False, does not overwrite, but checks that checksum is consistent.

    Returns
    -------
    file_path: string
        Full path of the created file.
    """
    download_path = os.path.join(data_home, remote.filename)
    if not os.path.exists(download_path) or force_overwrite:
        # If file doesn't exist or we want to overwrite, download it
        with DownloadProgressBar(
            unit='B', unit_scale=True, miniters=1, desc=remote.url.split('/')[-1]
        ) as t:
            try:
                _download_large_file(remote.url, download_path, t.update_to)
            except HTTPError:
                error_msg = """
                            mirdata failed to download the dataset!
                            Please try again in a few minutes.
                            If this error persists, please raise an issue at
                            https://github.com/mir-dataset-loaders/mirdata,
                            and tag it with 'broken-link'.
                            """
                raise HTTPError(error_msg)

    checksum = md5(download_path)
    if remote.checksum != checksum:
        raise IOError(
            '{} has an MD5 checksum ({}) '
            'differing from expected ({}), '
            'file may be corrupted.'.format(download_path, checksum, remote.checksum)
        )
    return download_path


def download_zip_file(zip_remote, save_dir, force_overwrite, cleanup=False):
    zip_download_path = download_from_remote(zip_remote, save_dir, force_overwrite)
    unzip(zip_download_path, save_dir, cleanup=cleanup)


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


def download_tar_file(tar_remote, save_dir, force_overwrite, cleanup=False):
    tar_download_path = download_from_remote(tar_remote, save_dir, force_overwrite)
    untar(tar_download_path, save_dir, cleanup=cleanup)


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
    if tar_path.endswith('tar.gz'):
        tfile = tarfile.open(tar_path, 'r:gz')
    else:
        tfile = tarfile.TarFile(tar_path, 'r')
    tfile.extractall(save_dir)
    tfile.close()
    if cleanup:
        os.remove(tar_path)
