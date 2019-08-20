# -*- coding: utf-8 -*-
"""RWC Popular Dataset Loader
"""
import csv
import librosa
import logging
import numpy as np
import os

import mirdata.utils as utils
import mirdata.download_utils as download_utils

# these functions are identical for all rwc datasets
from mirdata.rwc_classical import _load_beats, _load_sections

INDEX = utils.load_json_index('rwc_popular_index.json')
METADATA = None
METADATA_REMOTE = download_utils.RemoteFileMetadata(
    filename='rwc-p.csv',
    url='https://github.com/magdalenafuentes/metadata/archive/master.zip',
    checksum='7dbe87fedbaaa1f348625a2af1d78030',
)
DATASET_DIR = 'RWC-Popular'
ANNOTATIONS_REMOTE_1 = download_utils.RemoteFileMetadata(
    filename='AIST.RWC-MDB-P-2001.BEAT.zip',
    url='https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-P-2001.BEAT.zip',
    checksum='3858aa989535bd7196b3cd07b512b5b6',
)
ANNOTATIONS_REMOTE_2 = download_utils.RemoteFileMetadata(
    filename='AIST.RWC-MDB-P-2001.CHORUS.zip',
    url='https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-P-2001.CHORUS.zip',
    checksum='f76b3a32701fbd9bf78baa608f692a77',
)
ANNOTATIONS_REMOTE_3 = download_utils.RemoteFileMetadata(
    filename='AIST.RWC-MDB-P-2001.CHORD.zip',
    url='https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-P-2001.CHORD.zip',
    checksum='68379c88bc8ec3f1907b32a3579197c5',
)
ANNOTATIONS_REMOTE_4 = download_utils.RemoteFileMetadata(
    filename='AIST.RWC-MDB-P-2001.VOCA_INST.zip',
    url='https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-P-2001.VOCA_INST.zip',
    checksum='47ded648a496407ef49dba9c8bf80e87',
)


class Track(object):
    def __init__(self, track_id, data_home=None):
        if track_id not in INDEX:
            raise ValueError(
                '{} is not a valid track ID in RWC-Popular'.format(track_id)
            )

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)
        self._data_home = data_home

        self._track_paths = INDEX[track_id]

        if METADATA is None or METADATA['data_home'] != data_home:
            _reload_metadata(data_home)

        if METADATA is not None and track_id in METADATA:
            self._track_metadata = METADATA[track_id]
        else:
            # annotations with missing metadata
            self._track_metadata = {
                'piece_number': None,
                'suffix': None,
                'track_number': None,
                'title': None,
                'artist': None,
                'singer_information': None,
                'duration_sec': None,
                'tempo': None,
                'instruments': None,
                'drum_information': None,
            }

        self.audio_path = os.path.join(self._data_home, self._track_paths['audio'][0])

        self.piece_number = self._track_metadata['piece_number']
        self.suffix = self._track_metadata['suffix']
        self.track_number = self._track_metadata['track_number']
        self.title = self._track_metadata['title']
        self.artist = self._track_metadata['artist']
        self.singer_information = self._track_metadata['singer_information']
        self.duration_sec = self._track_metadata['duration_sec']
        self.tempo = self._track_metadata['tempo']
        self.instruments = self._track_metadata['instruments']
        self.drum_information = self._track_metadata['drum_information']

    def __repr__(self):
        repr_string = (
            "RWC-Popular Track(track_id={}, audio_path={}, "
            + "piece_number={}, suffix={}, track_number={}, title={}, "
            + "artist={}, singer_information={}, duration_sec={}, "
            + "tempo={}, instruments={}, drum_information={}, "
            + "sections=SectionData('start_times', 'end_times', 'sections'), "
            + "beats=BeatData('beat_times', 'beat_positions'))"
        )
        return repr_string.format(
            self.track_id,
            self.audio_path,
            self.piece_number,
            self.suffix,
            self.track_number,
            self.title,
            self.artist,
            self.singer_information,
            self.duration_sec,
            self.tempo,
            self.instruments,
            self.drum_information,
        )

    @utils.cached_property
    def sections(self):
        return _load_sections(
            os.path.join(self._data_home, self._track_paths['sections'][0])
        )

    @utils.cached_property
    def beats(self):
        return _load_beats(os.path.join(self._data_home, self._track_paths['beats'][0]))

    @utils.cached_property
    def chords(self):
        return _load_chords(
            os.path.join(self._data_home, self._track_paths['chords'][0])
        )

    @utils.cached_property
    def vocal_instrument_activity(self):
        return _load_voca_inst(
            os.path.join(self._data_home, self._track_paths['voca_inst'][0])
        )

    @property
    def audio(self):
        return librosa.load(self.audio_path, sr=None, mono=True)


def download(data_home=None, force_overwrite=False):

    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    annotations_path = os.path.join(data_home, 'annotations')

    download_utils.downloader(
        annotations_path,
        zip_downloads=[
            ANNOTATIONS_REMOTE_1,
            ANNOTATIONS_REMOTE_2,
            ANNOTATIONS_REMOTE_3,
            ANNOTATIONS_REMOTE_4,
        ],
        force_overwrite=force_overwrite,
    )

    info_message = """
        Unfortunately the audio files of the RWC-Popular dataset are not available
        for download. If you have the RWC-Popular dataset, place the contents into a
        folder called RWC-Popular with the following structure:
            > RWC-Popular/
                > annotations/
                > audio/rwc-p-m0i with i in [1 .. 7]
                > metadata-master/
        and copy the RWC-Popular folder to {}
    """.format(
        data_home
    )

    download_utils.downloader(
        data_home,
        zip_downloads=[METADATA_REMOTE],
        info_message=info_message,
        force_overwrite=force_overwrite,
    )


def validate(data_home=None, silence=False):
    """Validate if the stored dataset is a valid version

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        missing_files (list): List of file paths that are in the dataset index
            but missing locally
        invalid_checksums (list): List of file paths that file exists in the dataset
            index but has a different checksum compare to the reference checksum

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    missing_files, invalid_checksums = utils.validator(
        INDEX, data_home, silence=silence
    )
    return missing_files, invalid_checksums


def track_ids():
    """Return track ids

    Returns:
        (list): A list of track ids
    """
    return list(INDEX.keys())


def load(data_home=None, silence_validator=False):
    """Load RWC-Genre dataset

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    validate(data_home, silence=silence_validator)
    rwc_popular_data = {}
    for key in track_ids():
        rwc_popular_data[key] = Track(key, data_home=data_home)
    return rwc_popular_data


def _load_chords(chords_path):
    if not os.path.exists(chords_path):
        return None
    begs = []  # timestamps of chord beginnings
    ends = []  # timestamps of chord endings
    chords = []  # chord labels

    if os.path.exists(chords_path):
        with open(chords_path, 'r') as fhandle:
            reader = csv.reader(fhandle, delimiter='\t')
            for line in reader:
                begs.append(float(line[0]))
                ends.append(float(line[1]))
                chords.append(line[2])

    return utils.ChordData(np.array(begs), np.array(ends), np.array(chords))


def _load_voca_inst(voca_inst_path):
    if not os.path.exists(voca_inst_path):
        return None
    begs = []  # timestamps of vocal-instrument activity beginnings
    ends = []  # timestamps of vocal-instrument activity endings
    events = []  # vocal-instrument activity labels

    with open(voca_inst_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter='\t')
        raw_data = []
        for line in reader:
            if line[0] != 'Piece No.':
                raw_data.append(line)

    for i in range(len(raw_data)):
        # Parsing vocal-instrument activity as intervals (beg, end, event)
        if raw_data[i] != raw_data[-1]:
            begs.append(float(raw_data[i][0]))
            ends.append(float(raw_data[i + 1][0]))
            events.append(raw_data[i][1])

    return utils.EventData(np.array(begs), np.array(ends), np.array(events))


def _load_metadata(data_home):

    metadata_path = os.path.join(data_home, 'metadata-master', 'rwc-p.csv')

    if not os.path.exists(metadata_path):
        logging.info(
            'Metadata file {} not found.'.format(metadata_path)
            + 'You can download the metadata file by running download()'
        )
        return None

    with open(metadata_path, 'r') as fhandle:
        dialect = csv.Sniffer().sniff(fhandle.read(1024))
        fhandle.seek(0)
        reader = csv.reader(fhandle, dialect)
        raw_data = []
        for line in reader:
            if line[0] != 'Piece No.':
                raw_data.append(line)

    metadata_index = {}
    for line in raw_data:
        if line[0] == 'Piece No.':
            continue
        p = '00' + line[0].split('.')[1][1:]
        track_id = 'RM-P{}'.format(p[len(p) - 3 :])

        metadata_index[track_id] = {
            'piece_number': line[0],
            'suffix': line[1],
            'track_number': line[2],
            'title': line[3],
            'artist': line[4],
            'singer_information': line[5],
            'duration_sec': line[6],
            'tempo': line[7],
            'instruments': line[8],
            'drum_information': line[9],
        }

    metadata_index['data_home'] = data_home

    return metadata_index


def _reload_metadata(data_home):
    global METADATA
    METADATA = _load_metadata(data_home=data_home)


def cite():
    cite_data = """
===========  MLA ===========

If using beat and section annotations please cite:

Goto, Masataka, et al.,
"RWC Music Database: Popular, Classical and Jazz Music Databases.",
3rd International Society for Music Information Retrieval Conference (2002)

If using chord annotations please cite:

Cho, Taemin, and Juan P. Bello.,
"A feature smoothing method for chord recognition using recurrence plots.",
12th International Society for Music Information Retrieval Conference (2011)

If using vocal-instrument activity annotations please cite:

Mauch, Matthias, et al.,
"Timbre and Melody Features for the Recognition of Vocal Activity and Instrumental Solos in Polyphonic Music.",
12th International Society for Music Information Retrieval Conference (2011)

========== Bibtex ==========

If using beat and section annotations please cite:

@inproceedings{goto2002rwc,
  title={RWC Music Database: Popular, Classical and Jazz Music Databases.},
  author={Goto, Masataka and Hashiguchi, Hiroki and Nishimura, Takuichi and Oka, Ryuichi},
  booktitle={3rd International Society for Music Information Retrieval Conference},
  year={2002},
  series={ISMIR},
}

If using chord annotations please cite:

@inproceedings{cho2011feature,
  title={A feature smoothing method for chord recognition using recurrence plots},
  author={Cho, Taemin and Bello, Juan P},
  booktitle={12th International Society for Music Information Retrieval Conference},
  year={2011},
  series={ISMIR},
}

If using vocal-instrument activity annotations please cite:

@inproceedings{mauch2011timbre,
  title={Timbre and Melody Features for the Recognition of Vocal Activity and Instrumental Solos in Polyphonic Music.},
  author={Mauch, Matthias and Fujihara, Hiromasa and Yoshii, Kazuyoshi and Goto, Masataka},
  booktitle={ISMIR},
  year={2011},
  series={ISMIR},
}

"""
    print(cite_data)
