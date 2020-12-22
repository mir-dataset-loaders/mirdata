# -*- coding: utf-8 -*-
"""Utilities for downloading from the web.
"""

from collections import namedtuple
import os
from tqdm import tqdm
import urllib
import tarfile
import zipfile

from mirdata.validate import md5


class RemoteFileMetadata(object):
    """The metadata for a remote file

    Attributes:
        filename (str): the remote file's basename
        url (str): the remote file's url
        checksum (str): the remote file's md5 checksum
        destination_dir (str or None): the relative path for where to save the file

    """

    def __init__(self, filename, url, checksum, destination_dir):
        self.filename = filename
        self.url = url
        self.checksum = checksum
        self.destination_dir = destination_dir


def downloader(
    save_dir,
    remotes=None,
    partial_download=None,
    info_message=None,
    force_overwrite=False,
    cleanup=True,
):
    """Download data to `save_dir` and optionally print a message.

    Args:
        save_dir (str):
            The directory to download the data
        remotes (dict or None):
            A dictionary of RemoteFileMetadata tuples of data in zip format.
            If None, there is no data to download
        partial_download (list or None):
            A list of keys to partially download the remote objects of the download dict.
            If None, all data is downloaded
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

    if remotes is not None:
        if partial_download is not None:
            # check the keys in partial_download are in the download dict
            if not isinstance(partial_download, list) or any(
                [k not in remotes for k in partial_download]
            ):
                raise ValueError(
                    "partial_download must be a list which is a subset of {}".format(
                        remotes.keys()
                    )
                )
            objs_to_download = partial_download
        else:
            objs_to_download = list(remotes.keys())

        print("Starting to download {} to folder {}".format(objs_to_download, save_dir))

        for k in objs_to_download:
            print("> downloading {}".format(k))
            extension = os.path.splitext(remotes[k].filename)[-1]
            if ".zip" in extension:
                download_zip_file(remotes[k], save_dir, force_overwrite, cleanup)
            elif ".gz" in extension or ".tar" in extension or ".bz2" in extension:
                download_tar_file(remotes[k], save_dir, force_overwrite, cleanup)
            else:
                download_from_remote(remotes[k], save_dir, force_overwrite)

    if info_message is not None:
        print(info_message.format(save_dir))


class DownloadProgressBar(tqdm):
    """Wrap `tqdm` to show download progress"""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_from_remote(remote, save_dir, force_overwrite=False, clean_up=True):
    """Download a remote dataset into path
    Fetch a dataset pointed by remote's url, save into path using remote's
    filename and ensure its integrity based on the MD5 Checksum of the
    downloaded file.

    Adapted from scikit-learn's sklearn.datasets.base._fetch_remote.

    Args:
        remote (RemoteFileMetadata): Named tuple containing remote dataset
            meta information: url, filename and checksum
        save_dir (str): Directory to save the file to. Usually `data_home`
        force_overwrite  (bool):
            If True, overwrite existing file with the downloaded file.
            If False, does not overwrite, but checks that checksum is consistent.
        clean_up (bool):
            If True, overwrite existing file if the checksum can not be verified.

    Returns:
        file_path (str): Full path of the created file.
    """
    if remote.destination_dir is None:
        download_dir = save_dir
    else:
        download_dir = os.path.join(save_dir, remote.destination_dir)

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    download_path = os.path.join(download_dir, remote.filename)
    if (
        not os.path.exists(download_path)
        or force_overwrite
        or (
            clean_up
            and os.path.exists(download_path)
            and remote.checksum != md5(download_path)
        )
    ):
        # If file doesn't exist or we want to overwrite, download it
        with DownloadProgressBar(
            unit="B", unit_scale=True, unit_divisor=1024, miniters=1
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
                            mirdata failed to download the dataset from {}!
                            Please try again in a few minutes.
                            If this error persists, please raise an issue at
                            https://github.com/mir-dataset-loaders/mirdata,
                            and tag it with 'broken-link'.
                            """.format(
                    remote.url
                )
                print(error_msg)
                raise e

    checksum = md5(download_path)
    if remote.checksum != checksum:
        raise IOError(
            "{} has an MD5 checksum ({}) "
            "differing from expected ({}), "
            "file may be corrupted.".format(download_path, checksum, remote.checksum)
        )
    return download_path


def download_zip_file(zip_remote, save_dir, force_overwrite, cleanup=True):
    """Download and unzip a zip file.

    Args:
        zip_remote (RemoteFileMetadata):
            Object containing download information
        save_dir (str):
            Path to save downloaded file
        force_overwrite (bool):
            If True, overwrites existing files
        cleanup (bool):
            If True, remove zipfile after unziping. Default=False
    """
    zip_download_path = download_from_remote(
        zip_remote, save_dir, force_overwrite, cleanup
    )
    unzip(zip_download_path, cleanup=cleanup)


def extractall_unicode(zfile, out_dir):
    """Extract all files inside a zip archive to a output directory.
    In comparison to the zipfile, it checks for correct file name encoding

    Args:
        zfile (obj): Zip file object created with zipfile.ZipFile
        out_dir (str): Output folder

    """
    for m in zfile.infolist():
        data = zfile.read(m)  # extract zipped data into memory

        if m.filename.encode("cp437").decode() != m.filename.encode("utf8").decode():
            disk_file_name = os.path.join(out_dir, m.filename.encode("cp437").decode())
        else:
            disk_file_name = os.path.join(out_dir, m.filename)

        dir_name = os.path.dirname(disk_file_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        if not os.path.isdir(disk_file_name):
            with open(disk_file_name, "wb") as fd:
                fd.write(data)


def unzip(zip_path, cleanup=True):
    """Unzip a zip file inside it's current directory.

    Args:
        zip_path (str): Path to zip file
        cleanup (bool): If True, remove zipfile after unzipping. Default=False

    """
    zfile = zipfile.ZipFile(zip_path, "r")
    extractall_unicode(zfile, os.path.dirname(zip_path))
    zfile.close()
    if cleanup:
        os.remove(zip_path)


def download_tar_file(tar_remote, save_dir, force_overwrite, cleanup=True):
    """Download and untar a tar file.

    Args:
        tar_remote (RemoteFileMetadata): Object containing download information
        save_dir (str): Path to save downloaded file
        force_overwrite (bool): If True, overwrites existing files
        cleanup (bool): If True, remove tarfile after untarring. Default=False
    """
    tar_download_path = download_from_remote(
        tar_remote, save_dir, force_overwrite, cleanup
    )
    untar(tar_download_path, cleanup=cleanup)


def untar(tar_path, cleanup=True):
    """Untar a tar file inside it's current directory.

    Args:
        tar_path (str): Path to tar file
        cleanup (bool): If True, remove tarfile after untarring. Default=False
    """
    tfile = tarfile.open(tar_path, "r")
    tfile.extractall(os.path.dirname(tar_path))
    tfile.close()
    if cleanup:
        os.remove(tar_path)
