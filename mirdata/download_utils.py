# -*- coding: utf-8 -*-
"""functions for downloading from the web

Attributes:
    RemoteFileMetadata (namedtuple): It specifies the metadata of the remote file to download.
        The metadata consists of `filename`, `url`, `checksum`, and `destination_dir`.
"""

from collections import namedtuple
import os
from tqdm import tqdm
import urllib
import tarfile
import zipfile

from mirdata.utils import md5

# destination dir should be a relative path to save the file/s, or None
RemoteFileMetadata = namedtuple(
    'RemoteFileMetadata', ['filename', 'url', 'checksum', 'destination_dir']
)


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
                    'partial_download must be a list which is a subset of {}'.format(
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
            if '.zip' in extension:
                download_zip_file(remotes[k], save_dir, force_overwrite, cleanup)
            elif '.gz' in extension or '.tar' in extension:
                download_tar_file(remotes[k], save_dir, force_overwrite, cleanup)
            else:
                download_from_remote(remotes[k], save_dir, force_overwrite)

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

    Args:
        remote (RemoteFileMetadata): Named tuple containing remote dataset
            meta information: url, filename and checksum
        save_dir (str): Directory to save the file to. Usually `data_home`
        force_overwrite  (bool):
            If True, overwrite existing file with the downloaded file.
            If False, does not overwrite, but checks that checksum is consistent.

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
            '{} has an MD5 checksum ({}) '
            'differing from expected ({}), '
            'file may be corrupted.'.format(download_path, checksum, remote.checksum)
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
    zip_download_path = download_from_remote(zip_remote, save_dir, force_overwrite)
    unzip(zip_download_path, cleanup=cleanup)


def unzip(zip_path, cleanup=True):
    """Unzip a zip file inside it's current directory.

    Args:
        zip_path (str): Path to zip file
        cleanup (bool): If True, remove zipfile after unzipping. Default=False

    """
    zfile = zipfile.ZipFile(zip_path, 'r')
    zfile.extractall(os.path.dirname(zip_path))
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
    tar_download_path = download_from_remote(tar_remote, save_dir, force_overwrite)
    untar(tar_download_path, cleanup=cleanup)


def untar(tar_path, cleanup=True):
    """Untar a tar file inside it's current directory.

    Args:
        tar_path (str): Path to tar file
        cleanup (bool): If True, remove tarfile after untarring. Default=False
    """
    tfile = tarfile.open(tar_path, 'r')
    tfile.extractall(os.path.dirname(tar_path))
    tfile.close()
    if cleanup:
        os.remove(tar_path)
