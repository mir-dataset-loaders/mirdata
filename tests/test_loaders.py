# -*- coding: utf-8 -*-
from __future__ import absolute_import

import importlib
from inspect import signature
import io
import os
import sys

import mirdata

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
