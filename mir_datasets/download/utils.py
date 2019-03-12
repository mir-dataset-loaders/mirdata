from collections import namedtuple
import hashlib
import os
import tarfile
import urllib
import zipfile

from tqdm import tqdm

from .. import MIR_DATASETS_DIR

RemoteFileMetadata = namedtuple('RemoteFileMetadata',
                                ['filename', 'url', 'checksum'])


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as fhandle:
        for chunk in iter(lambda: fhandle.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_from_remote(remote, data_home=None):
    """Download a remote dataset into path
    Fetch a dataset pointed by remote's url, save into path using remote's
    filename and ensure its integrity based on the MD5 Checksum of the
    downloaded file.

    Adapted from scikit-learn's sklearn.datasets.base._fetch_remote.

    Parameters
    -----------
    remote : RemoteFileMetadata
        Named tuple containing remote dataset meta information: url, filename
        and checksum
    data_home : string
        Directory to save the file to.

    Returns
    -------
    file_path: string
        Full path of the created file.
    """
    # TODO: Check if download_path already exists?
    download_path = (
        os.path.join(MIR_DATASETS_DIR, remote.filename) if data_home is None
        else os.path.join(data_home, remote.filename)
    )

    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=remote.url.split('/')[-1]) as t:
        urllib.request.urlretrieve(
            remote.url, filename=download_path, reporthook=t.update_to)

    # urllib.request.urlretrieve(remote.url, download_path)
    checksum = md5(download_path)
    if remote.checksum != checksum:
        raise IOError("{} has an MD5 checksum ({}) "
                      "differing from expected ({}), "
                      "file may be corrupted.".format(download_path, checksum,
                                                      remote.checksum))
    return download_path


def unzip(zip_path, save_dir, cleanup=True):
    zfile = zipfile.ZipFile(zip_path, 'r')
    zfile.extractall(save_dir)
    zfile.close()
    if cleanup:
        os.remove(zip_path)


def untar(tar_path, save_dir, cleanup=True):
    tfile = tarfile.TarFile(tar_path, 'r')
    tfile.extractall(save_dir)
    tfile.close()
    if cleanup:
        os.remove(tar_path)
