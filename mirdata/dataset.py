"""Module containing the Dataset base class
"""
import importlib
import os
import random

from mirdata import download_utils
from mirdata import utils


class Dataset(object):
    """mirdata Dataset object

    Usage example:
    orchset = mirdata.Dataset('orchset')  # get the orchset dataset
    orchset.download()  # download orchset
    orchset.validate()  # validate orchset
    track = orchset.choice()  # load a random track
    print(track)  # see what data a track contains
    orchset.track_ids()  # load all track ids

    Attributes:
        dataset (str): the identifier of the dataset
        bibtex (str): dataset citation/s in bibtex format
        remotes (dict): data to be downloaded
        index (dict): dataset file index
        download_info (str): download instructions or caveats
        track_object (mirdata.track.Track): an uninstantiated Track object
        dataset_dir (str): dataset save folder
        readme (str): information about the dataset
        data_home (str): path where mirdata will look for the dataset

    """

    def __init__(self, dataset, data_home=None):
        """Inits a dataset by name and data location"""
        module = importlib.import_module("mirdata.{}".format(dataset))
        self.dataset = dataset
        self.bibtex = getattr(module, "BIBTEX", "No citation data provided")
        self.remotes = getattr(module, "REMOTES", {})
        self.index = module.DATA.index
        self.download_info = getattr(module, "DOWNLOAD_INFO", None)
        self.track_object = getattr(module, "Track", None)
        self.dataset_dir = module.DATASET_DIR
        self.download_fn = getattr(module, "download", download_utils.downloader)
        self.readme = module.__doc__

        if data_home is None:
            self.data_home = self.default_path
        else:
            self.data_home = data_home

    @property
    def default_path(self):
        """Get the default path for the dataset

        Returns:
            default_path (str): Local path to the dataset
        """
        mir_datasets_dir = os.path.join(os.getenv("HOME", "/tmp"), "mir_datasets")
        return os.path.join(mir_datasets_dir, self.dataset_dir)

    def track(self, track_id):
        """Load a track by track_id

        Args:
            track_id (str): track id of the track
        
        Returns:
            track (dataset.Track): an instance of this dataset's Track object
        """
        if self.track_object is None:
            raise NotImplementedError
        else:
            return self.track_object(track_id, self.data_home)

    def load_tracks(self):
        """Load all tracks in the dataset

        Returns:
            (dict): {`track_id`: track data}
        
        Raises:
            NotImplementedError: If the dataset does not support Track objects
        """
        return {self.track(track_id) for track_id in self.track_ids}

    def choice(self):
        """Choose a random track

        Returns:
            track (dataset.Track): a random Track object
        """
        return self.track(random.choice(self.track_ids))

    def cite(self):
        """Print the reference"""
        # TODO: use pybtex to convert to MLA
        print("========== BibTeX ==========")
        if isinstance(self.bibtex, str):
            print(self.bibtex)
        else:
            print("\n".join(self.bibtex.values()))

    def download(self, partial_download=None, force_overwrite=False, cleanup=True):
        """Download data to `save_dir` and optionally print a message.

        Args:
            partial_download (list or None):
                A list of keys of remotes to partially download.
                If None, all data is downloaded
            force_overwrite (bool):
                If True, existing files are overwritten by the downloaded files.
            cleanup (bool):
                Whether to delete the zip/tar file after extracting.

        Raises:
            ValueError: if invalid keys are passed to partial_download
            IOError: if a downloaded file's checksum is different from expected

        """
        self.download_fn(
            self.data_home,
            remotes=self.remotes,
            partial_download=partial_download,
            info_message=self.download_info,
            force_overwrite=force_overwrite,
            cleanup=cleanup,
        )

    @utils.cached_property
    def track_ids(self):
        """Return track ids

        Returns:
            (list): A list of track ids
        """
        return list(self.index.keys())

    def validate(self, verbose=True):
        """Validate if the stored dataset is a valid version

        Args:
            verbose (bool): If False, don't print output

        Returns:
            missing_files (list): List of file paths that are in the dataset index
                but missing locally
            invalid_checksums (list): List of file paths that file exists in the dataset
                index but has a different checksum compare to the reference checksum

        """
        missing_files, invalid_checksums = utils.validator(
            self.index, self.data_home, verbose=verbose
        )
        return missing_files, invalid_checksums

