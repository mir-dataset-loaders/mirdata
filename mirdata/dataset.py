import jams
import json
import os
import random
import re
import tqdm

from mirdata import download_utils
from mirdata import utils
from mirdata import track2


class Dataset(object):
    def __init__(self, module, data_home=None):
        self.name = module.name
        self.bibtex = module.bibtex
        self.remotes = module.remotes
        self.Track2 = module.Track2
        self.readme = module.__doc__
        self.module_path = module.__file__
        if data_home is None:
            self.data_home = self.dataset_default_path
        else:
            self.data_home = data_home

    # TODO: bikeshed the naming of the kwarg flag196
    def __getitem__(self, track_id, flag196="error"):
        track_id_info = self.Track2.parse_track_id(track_id)
        track_index = self.index[track_id].copy()
        for track_key in self.index[track_id]:
            if track_index[track_key][0] is not None:
                track_index[track_key][0] = os.path.join(
                    self.data_home, track_index[track_key][0]
                )
            else:
                del track_index[track_key]
        try:
            metadata = self.metadata
            if track_id in metadata:
                track_metadata = {**track_id_info, **metadata[track_id]}
            else:
                track_metadata = track_id_info
        except FileNotFoundError as file_not_found_error:
            # Failsafe mode
            if flag196 == "pass":
                track_metadata = track_id_info
            else:
                raise file_not_found_error
        return self.Track2(track_index, track_metadata, flag196=flag196)

    @property
    def dataset_default_path(self):
        mir_datasets_dir = os.path.join(os.getenv('HOME', '/tmp'), 'mir_datasets')
        dataset_dir = re.sub(' ', '-', self.name)
        return os.path.join(mir_datasets_dir, dataset_dir)

    @utils.cached_property
    def index(self):
        return self.load_index()

    @property
    def index_path(self):
        json_name = re.sub('[^0-9a-z]+', '_', self.name.lower()) + "_index.json"
        cwd = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(cwd, "indexes", json_name)

    @utils.cached_property
    def metadata(self):
        metadata_index = self.Track2.load_metadata(self.data_home)
        return metadata_index

    def choice(self):
        return self[random.choice(self.track_ids())]

    def cite(self):
        # TODO: use pybtex to convert to MLA
        print("========== BibTeX ==========")
        if isinstance(self.bibtex, str):
            print(self.bibtex)
        else:
            print("\n".join(self.bibtex.values()))

    def download(self, force_overwrite=False, cleanup=False, download_items=None):
        # TODO: perhaps find a shorter kwarg name than "download_items"?
        if not os.path.exists(self.data_home):
            os.makedirs(self.data_home, exist_ok=True)

        # By default, download all remotes
        if download_items is None:
            download_items = self.remotes

        # The list previus_remote_prints stores all the previously printed
        # messages. This avoids printing the same message over and over for
        # different remotes when multiple remotes are absent from download,
        # e.g. in ikala
        previous_remote_prints = []
        for remote_key in download_items:
            remote = self.remotes[remote_key]
            if isinstance(remote, str):
                if remote not in previous_remote_prints:
                    print(remote.format(self.data_home))
                previous_remote_prints.append(remote)
                continue

            if ".zip" in remote.url:
                download_utils.download_zip_file(
                    remote, self.data_home, force_overwrite, cleanup
                )
            elif ".tar.gz" in remote.url:
                download_utils.download_tar_file(
                    remote, self.data_home, force_overwrite, cleanup
                )
            else:
                download_utils.download_from_remote(
                    remote, self.data_home, force_overwrite
                )

    # TODO: bikeshed the naming of the kwarg flag196
    def load(self, verbose=False, flag196="error"):
        tracks = {}
        for track_id in tqdm.tqdm(self.index, disable=not verbose):
            tracks[track_id] = self.__getitem__(track_id, flag196=flag196)

        return tracks

    def load_index(self):
        with open(self.index_path) as f:
            return json.load(f)

    def track_ids(self):
        return list(self.index.keys())

    def validate(self, verbose=False):
        missing_files = []
        invalid_checksums = []
        for track_id in tqdm.tqdm(self.track_ids(), disable=not verbose):
            track_validation = self[track_id].validate()
            missing_files += track_validation[0]
            invalid_checksums += track_validation[1]
        return missing_files, invalid_checksums
