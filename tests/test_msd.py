# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import pytest

from mirdata import msd


def test_track():
    data_home = "tests/resources/mir_datasets/MSD"
    track = msd.Track("TRAAAFP128F931B4E3", data_home=data_home)

    with pytest.raises(ValueError):
        msd.Track("asdfasdf", data_home=data_home)

    assert track.metadata.songs[0].artist_name == "F.L.Y. (Fast Life Yungstaz)"
    assert track.analysis.songs[0].tempo == 141.968

    if sys.version_info[0] < 3:
        expected_artist_terms_str = "[u'rap', u'def jam']"
    else:
        expected_artist_terms_str = "['rap', 'def jam']"

    expected_str = "msd.Track(analysis=MSDAnalysis(len(bars_start)=145, len(beats_start)=582, songs=[MSDAnalysisSong(track_id=TRAAAFP128F931B4E3, duration=246.54322, key=3, [...])], [...]), metadata=MSDMetadata(songs=[MSDMetadataSong(artist_id=ARWNWOT1242077E494, artist_name=F.L.Y. (Fast Life Yungstaz), song_id=SOYKDDB12AB017EA7A, title=Bands, [...])], artist_terms={expected_artist_terms_str}, [...]), musicbrainz=MSDMusicBrainz(artist_mbtags=[], artist_mbtags_count=[], songs=[MSDMusicBrainzSong(year=0)]))".format(
        expected_artist_terms_str=expected_artist_terms_str
    )

    assert str(track) == expected_str


def test_track_ids():
    data_home = "tests/resources/mir_datasets/MSD"

    assert list(msd.track_ids(data_home=data_home)) == ["TRAAAFP128F931B4E3"]


def test_validate():
    data_home = "tests/resources/mir_datasets/MSD"
    missing_files, invalid_checksums = msd.validate(data_home=data_home, silence=True)

    # the test track, TRAAAFP128F931B4E3, is in the index deliberately
    index_sample_size = 10000
    num_test_tracks = 1
    assert len(missing_files) == index_sample_size - num_test_tracks
    assert len(invalid_checksums) == 0


def test_download(capsys):
    msd.download()
    captured = capsys.readouterr()
    assert "There are several ways of downloading the Million Song Dataset" in captured.out
