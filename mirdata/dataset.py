import jams
import json
import os
import tqdm
import re

from mirdata import utils


class Dataset(object):
    def __init__(
        self, name, load_track, load_metadata, bibtex, remotes, data_home=None
    ):
        super(Dataset, self).__init__()
        self.name = name
        self.load_metadata = load_metadata
        self.load_track = load_track
        self.bibtex = bibtex
        self.remotes = remotes
        if data_home is None:
            self.data_home = self.dataset_default_path
        else:
            self.data_home = data_home

    def __getitem__(self, track_id):
        track_metadata = self.metadata[track_id]
        track_index = self.index[track_id].copy()
        for track_key in self.index[track_id]:
            track_index[track_key][0] = os.path.join(
                self.data_home, track_index[track_key][0]
            )
        return Track(self.load_track, track_metadata, track_index)

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
        metadata_index = self.load_metadata(self.data_home)
        metadata_index['data_home'] = self.data_home
        return metadata_index

    def cite(self):
        bibtex_citation = self.bibtex
        print("========== BibTeX ==========")
        print(bibtex)

    def load(self, verbose=False):
        tracks = {}
        for track_id in tqdm.tqdm(self.index, disable=not verbose):
            tracks[track_id] = self[track_id]
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
