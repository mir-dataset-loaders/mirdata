import argparse
import csv
import hashlib
import json
import os
import itertools


acousticbrainz_genre_INDEX_PATH = '../mirdata/indexes/acousticbrainz_genre_index.json'
ACOUSTICBRAINZ_GENRE_ANNOTATION_SCHEMA = ['JAMS']


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
    with open(file_path, 'rb') as fhandle:
        for chunk in iter(lambda: fhandle.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def make_acousticbrainz_genre_index(data_path):
    count = 0
    acousticbrainz_genre_index = {}
    datasets = ['allmusic', 'lastfm', 'tagtraum', 'discogs']
    dataset_types = ['validation', 'train']
    for dataset, dataset_type in itertools.product(datasets, dataset_types):
        tsv_file = open(os.path.join(data_path, "acousticbrainz-mediaeval-" + dataset + "-" + dataset_type + ".tsv"))
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        next(read_tsv, None)
        for row in read_tsv:
            mbid = ""
            track_id = dataset + '#' + dataset_type
            for i, r in enumerate(row):
                track_id = track_id + '#' + r
                if i == 0:
                    mbid = r
            ann_path = os.path.join(data_path, "acousticbrainz-mediaeval-" + dataset_type, mbid[:2], mbid + ".json")
            acousticbrainz_genre_index[track_id] = {
                'annotations': (ann_path, md5(ann_path)),
            }
            print(count)
            count += 1
    with open(acousticbrainz_genre_INDEX_PATH, 'w') as fhandle:
        json.dump(acousticbrainz_genre_index, fhandle, indent=2)


def main(args):
    make_acousticbrainz_genre_index(args.acousticbrainz_genre_data_path)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Make acousticbrainz_genre index file.')
    PARSER.add_argument('acousticbrainz_genre_data_path', type=str, help='Path to acousticbrainz_genre data folder.')
    main(PARSER.parse_args())

