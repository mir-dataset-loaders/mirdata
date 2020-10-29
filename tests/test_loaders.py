# -*- coding: utf-8 -*-

import importlib
import inspect
from inspect import signature
import io
import os
import requests
import sys
import pytest

import mirdata
from mirdata import track
from tests.test_utils import DEFAULT_DATA_HOME

DATASETS = [importlib.import_module("mirdata.{}".format(d)) for d in mirdata.__all__]
CUSTOM_TEST_TRACKS = {
    'beatles': '0111',
    'giantsteps_key': '3',
    'dali': '4b196e6c99574dd49ad00d56e132712b',
    'giantsteps_tempo': '113',
    'guitarset': '03_BN3-119-G_solo',
    'medley_solos_db': 'd07b1fc0-567d-52c2-fef4-239f31c9d40e',
    'medleydb_melody': 'MusicDelta_Beethoven',
    'mridangam_stroke': '224030',
    'rwc_classical': 'RM-C003',
    'rwc_jazz': 'RM-J004',
    'rwc_popular': 'RM-P001',
    'salami': '2',
    'tinysol': 'Fl-ord-C4-mf-N-T14d',
}


def test_cite():
    for dataset in DATASETS:
        text_trap = io.StringIO()
        sys.stdout = text_trap
        dataset.cite()
        sys.stdout = sys.__stdout__


KNOWN_ISSUES = {}  # key is module, value is REMOTE key
DOWNLOAD_EXCEPTIONS = ["maestro"]


def test_download(mocker):
    for dataset in DATASETS:
        dataset_name = dataset.__name__.split(".")[1]

        # test parameters & defaults
        assert hasattr(dataset, "download"), "{} has no download method".format(
            dataset_name
        )
        assert hasattr(
            dataset.download, "__call__"
        ), "{}.download is not callable".format(dataset_name)
        params = signature(dataset.download).parameters
        assert (
            "data_home" in params
        ), "data_home must be an argument of {}.download".format(dataset_name)
        assert (
            params["data_home"].default is None
        ), "the default value of data_Home in {}.download should be None".format(
            dataset_name
        )

        # if there are no remotes, make sure partial_download,
        # force_overwrite, and cleanup are not parameters
        if not hasattr(dataset, "REMOTES"):
            assert (
                "partial_download" not in params
            ), "{} has no REMOTES, so its download method does not need a partial_download argument".format(
                dataset_name
            )
            assert (
                "force_overwrite" not in params
            ), "{} has no REMOTES so its download method does not need a force_overwrite argument".format(
                dataset_name
            )
            assert (
                "cleanup" not in params
            ), "{} has no REMOTES so its download method does not need a cleanup argument".format(
                dataset_name
            )
        # if there are remotes, make sure force_overwrite is specified and
        # the default is False
        else:
            assert (
                "force_overwrite" in params
            ), "{} has REMOTES, so its download method must have a force_overwrite parameter".format(
                dataset_name
            )
            assert (
                params["force_overwrite"].default is False
            ), "the force_overwrite parameter of {}.download must default to False".format(
                dataset_name
            )

            # if there are remotes but only one item, make sure partial_download
            # is not a parameter
            if len(dataset.REMOTES) == 1:
                assert (
                    "partial_download" not in params
                ), "{}.REMOTES has only one item, so its download method does not need a partial_download argument".format(
                    dataset_name
                )
            # if there is more than one item in remotes, make sure partial_download
            # is a parameter and the default is None
            else:
                assert (
                    "partial_download" in params
                ), "{}.REMOTES has multiple downloads, so its download method should have a partial_download argument".format(
                    dataset_name
                )
                assert (
                    params["partial_download"].default is None
                ), "the default argument of partial_download in {}.download should be None"

            extensions = [
                os.path.splitext(r.filename)[-1] for r in dataset.REMOTES.values()
            ]
            # if there are any zip or tar files to download, make sure cleanup
            # is a parameter and its default is True
            if any([e == ".zip" or e == ".gz" for e in extensions]):
                assert (
                    "cleanup" in params
                ), "{}.REMOTES contains zip or tar files, so its download method should have a cleanup argument".format(
                    dataset_name
                )
                assert (
                    params["cleanup"].default is True
                ), "the default value for cleanup in {}.download should be True".format(
                    dataset_name
                )
            # if there are no zip or tar files, make sure cleanup is not a parameter
            else:
                assert (
                    "cleanup" not in params
                ), "there are no zip or tar files in {}.REMOTES so its download method does not need a cleanup argument".format(
                    dataset_name
                )

        # check that the download method can be called without errors
        if hasattr(dataset, "REMOTES"):
            mock_downloader = mocker.patch.object(dataset, "REMOTES")
            if dataset_name not in DOWNLOAD_EXCEPTIONS:
                try:
                    dataset.download()
                except:
                    assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

                mocker.resetall()

            # check that links are online
            for key in dataset.REMOTES:
                # skip this test if it's in known issues
                if dataset_name in KNOWN_ISSUES and key in KNOWN_ISSUES[dataset_name]:
                    continue

                url = dataset.REMOTES[key].url
                try:
                    request = requests.head(url)
                    assert request.ok, "Link {} for {} does not return OK".format(
                        url, dataset_name
                    )
                except requests.exceptions.ConnectionError:
                    assert False, "Link {} for {} is unreachable".format(
                        url, dataset_name
                    )
                except:
                    assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])
        else:
            try:
                dataset.download()
            except:
                assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])


# This is magically skipped by the the remote fixture `skip_local` in conftest.py
# when tests are run with the --local flag
def test_validate(skip_local):
    for dataset in DATASETS:
        dataset_name = dataset.__name__.split(".")[1]
        data_home = os.path.join("tests/resources/mir_datasets", dataset.DATASET_DIR)
        try:
            dataset.validate(data_home=data_home)
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        try:
            dataset.validate(data_home=data_home, silence=True)
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        try:
            dataset.validate(data_home=None, silence=True)
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])


def test_load_and_trackids():
    for dataset in DATASETS:
        dataset_name = dataset.__name__.split(".")[1]
        try:
            track_ids = dataset.track_ids()
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        assert type(track_ids) is list, "{}.track_ids() should return a list".format(
            dataset_name
        )
        trackid_len = len(track_ids)

        data_home = os.path.join("tests/resources/mir_datasets", dataset.DATASET_DIR)
        try:
            dataset_data = dataset.load(data_home=data_home)
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        assert type(dataset_data) is dict, "{}.load should return a dictionary".format(
            dataset_name
        )
        assert (
            len(dataset_data.keys()) == trackid_len
        ), "the dictionary returned {}.load() does not have the same number of elements as {}.track_ids()".format(
            dataset_name, dataset_name
        )

        try:
            dataset_data_default = dataset.load()
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        assert (
            type(dataset_data_default) is dict
        ), "{}.load should return a dictionary".format(dataset_name)
        assert (
            len(dataset_data_default.keys()) == trackid_len
        ), "the dictionary returned {}.load() does not have the same number of elements as {}.track_ids()".format(
            dataset_name, dataset_name
        )


def test_track():
    data_home_dir = "tests/resources/mir_datasets"

    for dataset in DATASETS:

        dataset_name = dataset.__name__.split(".")[1]

        if dataset_name in CUSTOM_TEST_TRACKS:
            trackid = CUSTOM_TEST_TRACKS[dataset_name]
        else:
            trackid = dataset.track_ids()[0]

        try:
            track_default = dataset.Track(trackid)
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        assert track_default._data_home == os.path.join(
            DEFAULT_DATA_HOME, dataset.DATASET_DIR
        ), "{}: Track._data_home path is not set as expected".format(dataset_name)

        # test data home specified
        data_home = os.path.join(data_home_dir, dataset.DATASET_DIR)
        try:
            track_test = dataset.Track(trackid, data_home=data_home)
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        assert isinstance(
            track_test, track.Track
        ), "{}.Track must be an instance of type track.Track".format(dataset_name)

        assert hasattr(
            track_test, "to_jams"
        ), "{}.Track must have a to_jams method".format(dataset_name)

        # Validate JSON schema
        try:
            jam = track_test.to_jams()
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        assert jam.validate(), "Jams validation failed for {}.Track({})".format(
            dataset_name, trackid
        )

        # will fail if something goes wrong with __repr__
        try:
            text_trap = io.StringIO()
            sys.stdout = text_trap
            print(track_test)
            sys.stdout = sys.__stdout__
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        with pytest.raises(ValueError):
            dataset.Track("~faketrackid~?!")

        try:
            track_custom = dataset.Track(trackid, data_home="casa/de/data")
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        assert (
            track_custom._data_home == "casa/de/data"
        ), "{}: Track._data_home path is not set as expected".format(dataset_name)


# for load_* functions which require more than one argument
# module_name : {function_name: {parameter2: value, parameter3: value}}
EXCEPTIONS = {
    "dali": {"load_annotations_granularity": {"granularity": "notes"}},
    "guitarset": {
        "load_pitch_contour": {"string_num": 1},
        "load_note_ann": {"string_num": 1},
    },
}


def test_load_methods():
    for dataset in DATASETS:
        dataset_name = dataset.__name__.split(".")[1]

        all_methods = dir(dataset)
        load_methods = [
            getattr(dataset, m) for m in all_methods if m.startswith("load_")
        ]
        for load_method in load_methods:
            method_name = load_method.__name__
            params = [
                p
                for p in signature(load_method).parameters.values()
                if p.default == inspect._empty
            ]  # get list of parameters that don't have defaults

            # add to the EXCEPTIONS dictionary above if your load_* function needs
            # more than one argument.
            if dataset_name in EXCEPTIONS and method_name in EXCEPTIONS[dataset_name]:
                extra_params = EXCEPTIONS[dataset_name][method_name]
                with pytest.raises(IOError):
                    load_method("a/fake/filepath", **extra_params)
            else:
                with pytest.raises(IOError):
                    load_method("a/fake/filepath")


CUSTOM_TEST_MTRACKS = {}


def test_multitracks():
    data_home_dir = "tests/resources/mir_datasets"

    for dataset in DATASETS:

        dataset_name = dataset.__name__.split(".")[1]

        # TODO this is currently an opt-in test. Make it an opt out test
        # once #265 is addressed
        if dataset_name in CUSTOM_TEST_MTRACKS:
            mtrack_id = CUSTOM_TEST_MTRACKS[dataset_name]
        else:
            # there are no multitracks
            continue

        try:
            mtrack_default = dataset.MultiTrack(mtrack_id)
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        # test data home specified
        data_home = os.path.join(data_home_dir, dataset.DATASET_DIR)
        try:
            mtrack_test = dataset.MultiTrack(mtrack_id, data_home=data_home)
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        assert isinstance(
            mtrack_test, track.MultiTrack
        ), "{}.MultiTrack must be an instance of type track.MultiTrack".format(
            dataset_name
        )

        assert hasattr(
            mtrack_test, "to_jams"
        ), "{}.MultiTrack must have a to_jams method".format(dataset_name)

        # Validate JSON schema
        try:
            jam = mtrack_test.to_jams()
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        assert jam.validate(), "Jams validation failed for {}.Track({})".format(
            dataset_name, mtrack_id
        )
