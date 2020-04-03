# -*- coding: utf-8 -*-
from __future__ import absolute_import

import importlib
from inspect import signature
import io
import os
import sys
import pytest

import mirdata
import mirdata.track as track
from tests.test_utils import DEFAULT_DATA_HOME

DATASETS = [importlib.import_module("mirdata.{}".format(d)) for d in mirdata.__all__]


def test_cite():
    for dataset in DATASETS:
        text_trap = io.StringIO()
        sys.stdout = text_trap
        dataset.cite()
        sys.stdout = sys.__stdout__


def test_download():
    for dataset in DATASETS:
        assert hasattr(dataset, 'download')
        assert hasattr(dataset.download, '__call__')
        params = signature(dataset.download).parameters
        assert 'data_home' in params
        assert params['data_home'].default is None


def test_validate():
    for dataset in DATASETS:
        data_home = os.path.join('tests/resources/mir_datasets', dataset.DATASET_DIR)
        dataset.validate(data_home=data_home)
        dataset.validate(data_home=data_home, silence=True)
        dataset.validate(data_home=None, silence=True)


def test_load_and_trackids():
    for dataset in DATASETS:
        track_ids = dataset.track_ids()
        assert type(track_ids) is list
        trackid_len = len(track_ids)

        data_home = os.path.join('tests/resources/mir_datasets', dataset.DATASET_DIR)
        dataset_data = dataset.load(data_home=data_home)
        assert type(dataset_data) is dict
        assert len(dataset_data.keys()) == trackid_len

        dataset_data_default = dataset.load()
        assert type(dataset_data_default) is dict
        assert len(dataset_data_default.keys()) == trackid_len


def test_track():
    for dataset in DATASETS:
        print(str(dataset))
        trackid = dataset.track_ids()[0]

        # test data home None
        track_default = dataset.Track(trackid)
        assert track_default._data_home == os.path.join(
            DEFAULT_DATA_HOME, dataset.DATASET_DIR)

        assert isinstance(track_default, track.Track)

        assert hasattr(track_default, 'to_jams')

        # will fail if something goes wrong with __repr__
        print(track_default)

        with pytest.raises(ValueError):
            dataset.Track('~faketrackid~?!')

        track_custom = dataset.Track(trackid, data_home='casa/de/data')
        assert track_custom._data_home == 'casa/de/data'
