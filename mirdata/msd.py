# -*- coding: utf-8 -*-
"""Million Song Dataset Loader

The Million Song Dataset is a freely-available collection of audio features and metadata for a million contemporary popular music tracks.

More details can be found on http://millionsongdataset.com/

Attributes:
    DATASET_DIR (str): The directory name for Million Song Dataset. Set to `'msd'`.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tables

import mirdata.utils as utils
import mirdata.download_utils as download_utils

DATASET_DIR = 'MSD'


DATA = utils.LargeData('msd_index.json')


class Track(object):
    """MSD Track class

    Args:
        track_id (str): Track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        analysis (msd.MSDAnalysis): Echo Nest analysis
        metadata (msd.MSDMetadata): Artist and song metadata
        musicbrainz (msd.MSDMusicBrainz): MusicBrainz metadata
        """

    def __init__(self, track_id, data_home=None):
        self.track_id = track_id
        self.data_filename = _filename(track_id, data_home=data_home)

        if not os.path.exists(self.data_filename):
            raise ValueError('Invalid track ID: %s' % track_id)

        with tables.open_file(self.data_filename, mode='r') as f:
            self.analysis = MSDAnalysis(f.root.analysis)
            self.metadata = MSDMetadata(f.root.metadata)
            self.musicbrainz = MSDMusicBrainz(f.root.musicbrainz)

        self._data_home = data_home

    def __repr__(self):
        return (
            'msd.Track('
            + 'analysis={analysis}, '
            + 'metadata={metadata}, '
            + 'musicbrainz={musicbrainz})'
        ).format(
            analysis=self.analysis, metadata=self.metadata, musicbrainz=self.musicbrainz
        )


class MSDAnalysis(object):
    """MSD Analysis class

    Args:
        h5analysis (tables.Node): h5 object, instantiated in msd.Track

    Attributes:
        bars_confidence (list): confidence value (between 0 and 1) associated with each bar by The Echo Nest
        bars_start (list): start time of each bar according to The Echo Nest, this song has 99 bars
        beats_confidence (list): confidence value (between 0 and 1) associated with each beat by The Echo Nest
        beats_start (list): start time of each beat according to The Echo Nest, this song has 397 beats
        sections_confidence (list): confidence value (between 0 and 1) associated with each section by The Echo Nest
        sections_start (list): start time of each section according to The Echo Nest, this song has 10 sections
        segments_confidence (list): confidence value (between 0 and 1) associated with each segment by The Echo Nest
        segments_loudness_max (list): max loudness during each segment
        segments_loudness_max_time (list): time of the max loudness during each segment
        segments_loudness_start (list): loudness at the beginning of each segment
        segments_pitches (list): chroma features for each segment (normalized so max is 1.)
        segments_start (list): start time of each segment (~ musical event, or onset) according to The Echo Nest, this song has 935 segments
        segments_timbre (list): MFCC-like features for each segment
        songs (list): List of msd.MSDAnalysisSong
        tatums_confidence (list): confidence value (between 0 and 1) associated with each tatum by The Echo Nest
        tatums_start (list): start time of each tatum according to The Echo Nest, this song has 794 tatums
    """

    def __init__(self, h5analysis):
        self.bars_confidence = list(h5analysis['bars_confidence'])
        self.bars_start = list(h5analysis['bars_start'])
        self.beats_confidence = list(h5analysis['beats_confidence'])
        self.beats_start = list(h5analysis['beats_start'])
        self.sections_confidence = list(h5analysis['sections_confidence'])
        self.sections_start = list(h5analysis['sections_start'])
        self.segments_confidence = list(h5analysis['segments_confidence'])
        self.segments_loudness_max = list(h5analysis['segments_loudness_max'])
        self.segments_loudness_max_time = list(h5analysis['segments_loudness_max_time'])
        self.segments_loudness_start = list(h5analysis['segments_loudness_start'])
        self.segments_pitches = list(h5analysis['segments_pitches'])
        self.segments_start = list(h5analysis['segments_start'])
        self.segments_timbre = list(h5analysis['segments_timbre'])
        self.songs = [MSDAnalysisSong(s) for s in h5analysis['songs']]
        self.tatums_confidence = list(h5analysis['tatums_confidence'])
        self.tatums_start = list(h5analysis['tatums_start'])

    def __repr__(self):
        return (
            'MSDAnalysis('
            + 'len(bars_start)={len_bars_start}, '
            + 'len(beats_start)={len_beats_start}, '
            + 'songs={songs}, '
            + '[...])'
        ).format(
            len_bars_start=len(self.bars_start),
            len_beats_start=len(self.beats_start),
            songs=self.songs,
        )


class MSDAnalysisSong(object):
    """MSD Analysis class

    Args:
        h5analysis (tables.Node): h5 object, instantiated in msd.Track

    Attributes:
        analysis_sample_rate (int): Sample rate of analysis
        audio_md5 (str): Hash code of the audio used for the analysis by The Echo Nest
        danceability (float): Danceability measure of this song according to The Echo Nest (between 0 and 1, 0 => not analyzed)
        duration (float): Duration of the track in seconds
        end_of_fade_in (float): Time of the end of the fade in, at the beginning of the song, according to The Echo Nest
        energy (float): Energy measure (not in the signal processing sense) according to The Echo Nest (between 0 and 1, 0 => not analyzed)
        key (int): Estimation of the key the song is in by The Echo Nest
        key_confidence (float): Confidence of the key estimation
        loudness (float): General loudness of the track
        mode (int): Estimation of the mode the song is in by The Echo Nest
        mode_confidence (float): Confidence of the mode estimation
        start_of_fade_out (float): Start time of the fade out, in seconds, at the end of the song, according to The Echo Nest
        tempo (float): Tempo in BPM according to The Echo Nest
        time_signature (int): Time signature of the song according to The Echo Nest, i.e. usual number of beats per bar
        time_signature_confidence (float): Confidence of the time signature estimation
        track_id (str): Artist_mbid: db92a151-1ac2-438b-bc43-b82e149ddd50
        idx_* (int): Array indices
    """

    def __init__(self, h5song):
        self.analysis_sample_rate = h5song['analysis_sample_rate']
        self.audio_md5 = _string(h5song['audio_md5'])
        self.danceability = h5song['danceability']
        self.duration = h5song['duration']
        self.end_of_fade_in = h5song['end_of_fade_in']
        self.energy = h5song['energy']
        self.key = h5song['key']
        self.key_confidence = h5song['key_confidence']
        self.loudness = h5song['loudness']
        self.mode = h5song['mode']
        self.mode_confidence = h5song['mode_confidence']
        self.start_of_fade_out = h5song['start_of_fade_out']
        self.tempo = h5song['tempo']
        self.time_signature = h5song['time_signature']
        self.time_signature_confidence = h5song['time_signature_confidence']
        self.track_id = _string(h5song['track_id'])
        self.idx_bars_confidence = h5song['idx_bars_confidence']
        self.idx_bars_start = h5song['idx_bars_start']
        self.idx_beats_confidence = h5song['idx_beats_confidence']
        self.idx_beats_start = h5song['idx_beats_start']
        self.idx_sections_confidence = h5song['idx_sections_confidence']
        self.idx_sections_start = h5song['idx_sections_start']
        self.idx_segments_confidence = h5song['idx_segments_confidence']
        self.idx_segments_loudness_max = h5song['idx_segments_loudness_max']
        self.idx_segments_loudness_max_time = h5song['idx_segments_loudness_max_time']
        self.idx_segments_loudness_start = h5song['idx_segments_loudness_start']
        self.idx_segments_pitches = h5song['idx_segments_pitches']
        self.idx_segments_start = h5song['idx_segments_start']
        self.idx_segments_timbre = h5song['idx_segments_timbre']
        self.idx_tatums_confidence = h5song['idx_tatums_confidence']
        self.idx_tatums_start = h5song['idx_tatums_start']

    def __repr__(self):
        return (
            'MSDAnalysisSong('
            + 'track_id={track_id}, '
            + 'duration={duration}, '
            + 'key={key}, '
            + '[...])'
        ).format(track_id=self.track_id, duration=self.duration, key=self.key,)


class MSDMetadata(object):
    """MSD Metadata class

    Args:
        h5metadata (tables.Node): h5 object, instantiated in msd.Track

    Attributes:
        artist_terms (list): This artist has 12 terms (tags) from The Echo Nest
        artist_terms_freq (list): Frequency of the 12 terms from The Echo Nest (number between 0 and 1)
        artist_terms_weight (list): Weight of the 12 terms from The Echo Nest (number between 0 and 1)
        similar_artists (list): A list of 100 artists (their Echo Nest ID) similar to Rick Astley according to The Echo Nest
        songs (list): List of msd.MSDMetadataSong
    """

    def __init__(self, h5metadata):
        self.artist_terms = _string_list(h5metadata['artist_terms'])
        self.artist_terms_freq = list(h5metadata['artist_terms_freq'])
        self.artist_terms_weight = list(h5metadata['artist_terms_weight'])
        self.similar_artists = _string_list(h5metadata['similar_artists'])
        self.songs = [MSDMetadataSong(s) for s in h5metadata['songs']]

    def __repr__(self):
        return (
            'MSDMetadata('
            + 'songs={songs}, '
            + 'artist_terms={artist_terms}, '
            + '[...])'
        ).format(songs=self.songs, artist_terms=self.artist_terms)


class MSDMetadataSong(object):
    """MSD Metadata class

    Args:
        h5metadata (tables.Node): h5 object, instantiated in msd.Track

    Attributes:
        analyzer_version (str): Version of the Echo Nest analyzer that produced these attributes
        artist_7digitalid (int): The ID of the artist on the service 7digital.com
        artist_familiarity (float): Algorithmic estimation of the familiarity of the artist
        artist_hotttnesss (float): Algorithmic estimation of the 'hotttnesss' of the artist
        artist_id (str): Echo Nest artist ID
        artist_latitude (float): Latitude of the artist location
        artist_location (str): Artist location name
        artist_longitude (float): Longitude of the artist location
        artist_mbid (str): ID from musicbrainz.org
        artist_name (str): Name of the artist
        artist_playmeid (int): ID from playme.com, or -1
        genre (str): Primary genre
        release (str): Album name
        release_7digitalid (int): release 7digitalid  int  ID from 7digital.com or -1
        song_hotttnesss (float): Algorithmic estimation of the 'hotttnesss' of the song
        song_id (str): Echo Nest song ID
        title (str): Song titlen
        track_7digitalid (int): ID from 7digital.com or -1
        idx_* (int): Array indices
    """

    def __init__(self, h5song):
        self.analyzer_version = _string(h5song['analyzer_version'])
        self.artist_7digitalid = h5song['artist_7digitalid']
        self.artist_familiarity = h5song['artist_familiarity']
        self.artist_hotttnesss = h5song['artist_hotttnesss']
        self.artist_id = _string(h5song['artist_id'])
        self.artist_latitude = h5song['artist_latitude']
        self.artist_location = _string(h5song['artist_location'])
        self.artist_longitude = h5song['artist_longitude']
        self.artist_mbid = _string(h5song['artist_mbid'])
        self.artist_name = _string(h5song['artist_name'])
        self.artist_playmeid = h5song['artist_playmeid']
        self.genre = _string(h5song['genre'])
        self.release = _string(h5song['release'])
        self.release_7digitalid = h5song['release_7digitalid']
        self.song_hotttnesss = h5song['song_hotttnesss']
        self.song_id = _string(h5song['song_id'])
        self.title = _string(h5song['title'])
        self.track_7digitalid = h5song['track_7digitalid']
        self.idx_artist_terms = h5song['idx_artist_terms']
        self.idx_similar_artists = h5song['idx_similar_artists']

    def __repr__(self):
        return (
            'MSDMetadataSong('
            + 'artist_id={artist_id}, '
            + 'artist_name={artist_name}, '
            + 'song_id={song_id}, '
            + 'title={title}, [...])'
        ).format(
            artist_id=self.artist_id,
            artist_name=self.artist_name,
            song_id=self.song_id,
            title=self.title,
        )


class MSDMusicBrainz(object):
    """MSD MusicBrainz class

    Args:
        h5mb (tables.Node): h5 object, instantiated in msd.Track

    Attributes:
        artist_mbtags (list): Tags from musicbrainz.org
        artist_mbtags_count (int): Number of MusicBrainz tags
        songs (list): List of msd.MSDMusicBrainzSong objects
    """

    def __init__(self, h5mb):
        self.artist_mbtags = _string_list(h5mb['artist_mbtags'])
        self.artist_mbtags_count = list(h5mb['artist_mbtags_count'])
        self.songs = [MSDMusicBrainzSong(s) for s in h5mb['songs']]

    def __repr__(self):
        return (
            'MSDMusicBrainz('
            + 'artist_mbtags={artist_mbtags}, '
            + 'artist_mbtags_count={artist_mbtags_count}, '
            + 'songs={songs})'
        ).format(
            artist_mbtags=self.artist_mbtags,
            artist_mbtags_count=self.artist_mbtags_count,
            songs=self.songs,
        )


class MSDMusicBrainzSong(object):
    """MSD MusicBrainz song class

    Args:
        h5mb (tables.Node): h5 object, instantiated in msd.Track

    Attributes:
        year (int): Release year of the song
        idx_artist_mbtags (int): Array index
    """

    def __init__(self, h5song):
        self.year = h5song['year']
        self.idx_artist_mbtags = h5song['idx_artist_mbtags']

    def __repr__(self):
        return 'MSDMusicBrainzSong(year={year})'.format(year=self.year)


def _string_list(lst):
    return [_string(x) for x in lst]


def _string(s):
    return s.decode('utf-8')


def _filename(track_id, data_home=None):
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    # format: 'data/J/D/V/TRJDVIB12903CF9F35.h5'
    return os.path.join(
        data_home, 'data', track_id[2], track_id[3], track_id[4], track_id + '.h5'
    )


def track_ids(data_home=None):
    """Generator that yields all MSD track IDs

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (generator) Track IDs
    """

    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    for _, _, files in os.walk(data_home):
        for name in files:
            if name.endswith('.h5'):
                yield name.replace('.h5', '')


def validate(data_home=None, silence=False):
    """Validate if the stored dataset is a valid version.

    Since the MSD is very large, only a subset of 10000 tracks (1% of the dataset) is validated

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
        DATA.index, data_home, silence=silence
    )
    return missing_files, invalid_checksums


def download(data_home=None):
    """The MSD is not available for downloading directly.
    This function prints a helper message to set up an EC2 instance
    with the MSD mounted

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    """

    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    info_message = """
There are several ways of downloading the Million Song Dataset,
listed on http://millionsongdataset.com/pages/getting-dataset/.

The easiest way is to create an instance on Amazon Web Services (AWS),
and do the analysis there. First sign up https://aws.amazon.com/
(requires entering credit card details).

Then, in a on your local machine:

# Install the AWS command line interface
$ pip install awscli

# Configure AWS
#   - Create a new access key on https://console.aws.amazon.com/iam/home?region=us-east-1#/security_credentials
#   - Enter the access key ID and secret access key
#   - Set zone to us-east-1
aws configure  # zone=us-east-1 and credentials

# Create a new SSH key pair
$ KEY_NAME=my-aws-keypair
$ PEM_PATH=~/.ssh/aws-key-pair.pem
$ aws ec2 create-key-pair --key-name $KEY_NAME --query 'KeyMaterial' --output text > $PEM_PATH

# Create a Ubuntu EC2 instance
$ UBUNTU_1804=ami-04b9e92b5572fa0d1
$ aws ec2 run-instances --image-id $UBUNTU_1804 --count 1 --key-pair $KEY_NAME --instance-type m1.small > /tmp/instance.json

# Create a volume from the MSD snapshot
$ AVAILABILITY_ZONE=$(jq -r '.Instances[0].Placement.AvailabilityZone' /tmp/instance.json)
$ MSD_SNAPSHOT=snap-5178cf30
$ aws ec2 create-volume --availability-zone $AVAILABILITY_ZONE --snapshot-id $MSD_SNAPSHOT --volume-type gp2 > /tmp/volume.json

# Attach the volume to the instance
$ VOLUME_ID=$(jq -r '.VolumeId' /tmp/volume.json)
$ INSTANCE_ID=$(jq -r '.Instances[0].InstanceId' /tmp/instance.json)
$ DEVICE=/dev/xvdk
$ aws ec2 attach-volume --volume-id $VOLUME_ID --instance-id $INSTANCE_ID --device $DEVICE

# Connect to the instance
$ INSTANCE_HOSTNAME=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID | jq -r .Reservations[0].Instances[0].PublicDnsName)
$ MSD_FOLDER={msd_folder}
$ ssh -A -i $PEM_PATH ubuntu@$INSTANCE_HOSTNAME

# On the instance, mount the MSD:
$ sudo mkdir $MSD_FOLDER
$ sudo mount $DEVICE $MSD_FOLDER
$ sudo chmod 777 $MSD_FOLDER'

""".format(
        msd_folder=data_home
    )

    download_utils.downloader(data_home, info_message=info_message)


def cite():
    """Print the reference"""

    cite_data = """
=========== MLA ===========
Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere.
The Million Song Dataset. In Proceedings of the 12th International Society
for Music Information Retrieval Conference (ISMIR 2011), 2011.
========== Bibtex ==========
@inproceedings{Bertin-Mahieux2011,
  author = {Bertin-Mahieux, Thierry and Ellis, Daniel P.W. and Whitman, Brian and Lamere, Paul},
  title = {The Million Song Dataset},
  booktitle = {{Proceedings of the 12th International Conference on Music Information
        Retrieval ({ISMIR} 2011)}},
  year = {2011},
}
"""
    print(cite_data)
