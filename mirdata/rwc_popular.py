"""RWC Popular Dataset Loader
"""
from collections import namedtuple
import csv
import numpy as np
import os

import mirdata.utils as utils

RWC_POPULAR_INDEX = utils.load_json_index("rwc_popular_index.json")
RWC_POPULAR_METADATA = utils.RemoteFileMetadata(
    filename='rwc-mdb-j.html',
    url='view-source:https://staff.aist.go.jp/m.goto/RWC-MDB/rwc-mdb-j.html',
    checksum=None)
RWC_POPULAR_DIR = 'RWC-Popular'
RWC_POPULAR_ANNOT_REMOTE_1 = utils.RemoteFileMetadata(
    filename='AIST.RWC-MDB-P-2001.BEAT.zip',
    url='https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-P-2001.BEAT.zip',
    checksum='3858aa989535bd7196b3cd07b512b5b6')
RWC_POPULAR_ANNOT_REMOTE_2 =  utils.RemoteFileMetadata(
    filename='AIST.RWC-MDB-P-2001.CHORUS.zip',
    url='https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-P-2001.CHORUS.zip',
    checksum='f76b3a32701fbd9bf78baa608f692a77')


RWCPopularTrack = namedtuple(
    'RWCPopularTrack',
    ['track_id',
     'audio_path',
     'sections',
     'beats',
     'duration_sec',
     'title',
     'artist',
     'variation',
     'instruments']
)


def download(data_home=None, clobber=False):
    save_path = utils.get_save_path(data_home)
    dataset_path = os.path.join(save_path, RWC_POPULAR_DIR, 'annotations')

    if clobber:
        utils.clobber_all(RWC_POPULAR_ANNOT_REMOTE_1,
                          dataset_path,
                          data_home)
        utils.clobber_all(RWC_POPULAR_ANNOT_REMOTE_2,
                          dataset_path,
                          data_home)

    if utils.check_validated(dataset_path):
        print("""
                The {} dataset has already been downloaded and validated.
                Skipping download of dataset. If you feel this is a mistake please
                rerun and set clobber to true
                """.format(RWC_POPULAR_DIR))
        return

    download_path_1 = utils.download_from_remote(
        RWC_POPULAR_ANNOT_REMOTE_1, data_home=data_home, clobber=clobber)
    download_path_2 = utils.download_from_remote(
        RWC_POPULAR_ANNOT_REMOTE_2, data_home=data_home, clobber=clobber)

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    utils.unzip(download_path_1, dataset_path, cleanup=True)
    utils.unzip(download_path_2, dataset_path, cleanup=True)

    missing_files, invalid_checksums = validate(dataset_path, data_home)
    if missing_files or invalid_checksums:
        print("""
            Unfortunately the audio files of the RWC-Popular dataset are not available
            for download. If you have the RWC-Popular dataset, place the contents into a
            folder called RWC-Popular with the following structure:
                > RWC-Popular/
                    > annotations/
                    > audio/
            and copy the RWC-Popular folder to {}
        """.format(save_path))

    # metadata
    # utils.download_from_remote(
    #         RWC_POPULAR_METADATA, data_home=data_home, clobber=clobber)


def validate(dataset_path, data_home=None):
    missing_files, invalid_checksums = utils.validator(RWC_POPULAR_INDEX, data_home, dataset_path)
    return missing_files, invalid_checksums


def track_ids():
    return list(RWC_POPULAR_INDEX.keys())


def load(data_home=None):
    save_path = utils.get_save_path(data_home)
    dataset_path = os.path.join(save_path, RWC_POPULAR_DIR, 'annotations')
    validate(dataset_path, data_home)
    rwc_popular_data = {}
    for key in track_ids():
        rwc_popular_data[key] = load_track(key, data_home=data_home)
    return rwc_popular_data


def load_track(track_id, data_home=None):
    if track_id not in RWC_POPULAR_INDEX.keys():
        raise ValueError(
            "{} is not a valid track ID in RWC_Popular".format(track_id))
    track_data = RWC_POPULAR_INDEX[track_id]

    # if RWC_POPULAR_METADATA is None or RWC_POPULAR_METADATA['data_home'] != data_home:
    #     _reload_metadata(data_home)
    #     if RWC_POPULAR_METADATA is None:
    #         raise EnvironmentError("Could not find RWC_Popular metadata file")

    # if track_id in RWC_POPULAR_METADATA.keys():
    #     track_metadata = RWC_POPULAR_METADATA[track_id]
    # else:
    #     # annotations with missing metadata
    track_metadata = {
        'duration_sec': None, 'title': None, 'artist': None,
        'variation': None, 'instruments': None
    }
    rwc_popular_path = utils.get_local_path(data_home, RWC_POPULAR_DIR)
    annotations_dir = os.path.join(rwc_popular_path, 'annotations')
    sections = _load_sections(annotations_dir, track_id)
    beats = _load_beats(annotations_dir, track_id)

    return RWCPopularTrack(
        track_id,
        utils.get_local_path(data_home, track_data['audio'][0]),
        sections,
        beats,
        track_metadata['duration_sec'],
        track_metadata['title'],
        track_metadata['artist'],
        track_metadata['variation'],
        track_metadata['instruments'],
    )

def _load_sections(sections_path, track_id):
    begs, ends, secs = [], [], []
    file_path = os.path.join(sections_path, 'AIST.RWC-MDB-P-2001.{}'.format('CHORUS'),
                             '{}.{}.TXT'.format(track_id, 'CHORUS'))
    if os.path.exists(file_path):
        with open(file_path, 'r') as fhandle:
                reader = csv.reader(fhandle, delimiter='\t')
                for line in reader:
                    begs.append(float(line[0])/100)
                    ends.append(float(line[1])/100)
                    secs.append(line[2])
        begs, ends, secs = np.array(begs), np.array(ends), np.array(secs)
        # # remove sections with length == 0
        # times_revised = np.delete(
        #     times, np.where(np.diff(times) == 0))
        # secs_revised = np.delete(
        #     secs, np.where(np.diff(times) == 0))
        data = utils.SectionData(begs,
                          ends,
                          secs)
    else:
        data = None

    return data


def _load_beats(beats_path, track_id):
    pass

def _load_metadata(data_home):

    pass
    # metadata_path = utils.get_local_path(
    #     data_home, os.path.join(
    #         RWC_POPULAR_DIR, 'rwc_popular-data-public-master', 'metadata', 'metadata.csv'
    #     )
    # )
    #
    # if not os.path.exists(metadata_path):
    #     return None
    #
    # with open(metadata_path, 'r') as fhandle:
    #     reader = csv.reader(fhandle, delimiter=',')
    #     raw_data = []
    #     for line in reader:
    #         if line[0] == 'SONG ID':
    #             continue
    #         raw_data.append(line)
    #
    # metadata_index = {}
    # for line in raw_data:
    #     track_id = line[0]
    #
    #     metadata_index[track_id] = {
    #         'source': line[1],
    #         'annotator_1_id': line[2],
    #         'annotator_2_id': line[3],
    #         'duration_sec': line[5],
    #         'title': line[7],
    #         'artist': line[8],
    #         'annotator_1_time': line[10],
    #         'annotator_2_time': line[11],
    #         'class': line[14],
    #         'popular': line[15],
    #     }

    # metadata_index['data_home'] = data_home

    # return metadata_index


def _reload_metadata(data_home):
    global RWC_POPULAR_METADATA
    RWC_POPULAR_METADATA = _load_metadata(data_home=data_home)


def cite():
    cite_data = """
===========  MLA ===========

Goto, Masataka, et al., 
"RWC Music Database: Popular, Classical and Jazz Music Databases.",
3rd International Society for Music Information Retrieval Conference (2002)

========== Bibtex ==========

@inproceedings{goto2002rwc,
  title={RWC Music Database: Popular, Classical and Jazz Music Databases.},
  author={Goto, Masataka and Hashiguchi, Hiroki and Nishimura, Takuichi and Oka, Ryuichi},
  booktitle={3rd International Society for Music Information Retrieval Conference},
  year={2002},
  series={ISMIR},
}

"""

    print(cite_data)
