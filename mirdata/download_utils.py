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
from tqdm import tqdm
from six.moves import urllib
import six
import sys

from mirdata.utils import md5


# destination dir should be a relative path to save the file/s, or None
RemoteFileMetadata = namedtuple(
    'RemoteFileMetadata', ['filename', 'url', 'checksum', 'destination_dir']
)


def downloader(
    save_dir,
    zip_downloads=None,
    tar_downloads=None,
    file_downloads=None,
    info_message=None,
    force_overwrite=False,
    cleanup=False,
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
        cleanup (bool):
            Whether to delete the zip/tar file after extracting.

    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if zip_downloads is not None:
        for zip_download in zip_downloads:
            download_zip_file(zip_download, save_dir, force_overwrite, cleanup)

    if tar_downloads is not None:
        for tar_download in tar_downloads:
            download_tar_file(tar_download, save_dir, force_overwrite, cleanup)

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


def download_from_remote(remote, save_dir, force_overwrite=False):
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
    save_dir: string
        Directory to save the file to. Usually `data_home`
    force_overwrite: bool
        If True, overwrite existing file with the downloaded file.
        If False, does not overwrite, but checks that checksum is consistent.

    Returns
    -------
    file_path: string
        Full path of the created file.
    """
    if remote.destination_dir is None:
        download_dir = save_dir
    else:
        download_dir = os.path.join(save_dir, remote.destination_dir)

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    download_path = os.path.join(download_dir, remote.filename)
    if not os.path.exists(download_path) or force_overwrite:
        # If file doesn't exist or we want to overwrite, download it
        with DownloadProgressBar(
            unit='B', unit_scale=True, unit_divisor=1024, miniters=1
        ) as t:
            try:
                urllib.request.urlretrieve(
                    remote.url,
                    filename=download_path,
                    reporthook=t.update_to,
                    data=None,
                )
            except Exception as e:
                error_msg = """
                            mirdata failed to download the dataset!
                            Please try again in a few minutes.
                            If this error persists, please raise an issue at
                            https://github.com/mir-dataset-loaders/mirdata,
                            and tag it with 'broken-link'.
                            """
                print(error_msg)
                raise e

    checksum = md5(download_path)
    if remote.checksum != checksum:
        raise IOError(
            '{} has an MD5 checksum ({}) '
            'differing from expected ({}), '
            'file may be corrupted.'.format(download_path, checksum, remote.checksum)
        )
    return download_path


def download_zip_file(zip_remote, save_dir, force_overwrite, cleanup=False):
    """Download and unzip a zip file.

    Parameters
    ----------
    zip_remote: RemoteFileMetadata
        Object containing download information
    save_dir: str
        Path to save downloaded file
    force_overwrite: bool
        If True, overwrites existing files
    cleanup: bool, default=False
        If True, remove zipfile after unziping.
    """
    zip_download_path = download_from_remote(zip_remote, save_dir, force_overwrite)
    unzip(zip_download_path, cleanup=cleanup)


def unzip(zip_path, cleanup=False):
    """Unzip a zip file inside it's current directory.

    Parameters
    ----------
    zip_path: str
        Path to zip file
    cleanup: bool, default=False
        If True, remove zipfile after unzipping.
    """
    zfile = zipfile.ZipFile(zip_path, 'r')
    zfile.extractall(os.path.dirname(zip_path))
    zfile.close()
    if cleanup:
        os.remove(zip_path)


def download_tar_file(tar_remote, save_dir, force_overwrite, cleanup=False):
    """Download and untar a tar file.

    Parameters
    ----------
    tar_remote: RemoteFileMetadata
        Object containing download information
    save_dir: str
        Path to save downloaded file
    force_overwrite: bool
        If True, overwrites existing files
    cleanup: bool, default=False
        If True, remove tarfile after untarring.
    """
    tar_download_path = download_from_remote(tar_remote, save_dir, force_overwrite)
    untar(tar_download_path, cleanup=cleanup)


def untar(tar_path, cleanup=False):
    """Untar a tar file inside it's current directory.

    Parameters
    ----------
    tar_path: str
        Path to tar file
    cleanup: bool, default=False
        If True, remove tarfile after untarring.
    """
    tfile = tarfile.open(tar_path, 'r')
    tfile.extractall(os.path.dirname(tar_path))
    tfile.close()
    if cleanup:
        os.remove(tar_path)
