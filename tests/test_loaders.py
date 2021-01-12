# -*- coding: utf-8 -*-

import importlib
import inspect
from inspect import signature
import io
import os
import sys
import pytest
import requests


import mirdata
from mirdata import core, download_utils
from tests.test_utils import DEFAULT_DATA_HOME

DATASETS = mirdata.DATASETS
CUSTOM_TEST_TRACKS = {
    "beatles": "0111",
    "cante100": "008",
    "giantsteps_key": "3",
    "dali": "4b196e6c99574dd49ad00d56e132712b",
    "giantsteps_tempo": "113",
    "guitarset": "03_BN3-119-G_solo",
    "irmas": "1",
    "medley_solos_db": "d07b1fc0-567d-52c2-fef4-239f31c9d40e",
    "medleydb_melody": "MusicDelta_Beethoven",
    "mridangam_stroke": "224030",
    "rwc_classical": "RM-C003",
    "rwc_jazz": "RM-J004",
    "rwc_popular": "RM-P001",
    "salami": "2",
    "saraga_carnatic": "116_Bhuvini_Dasudane",
    "saraga_hindustani": "59_Bairagi",
    "tinysol": "Fl-ord-C4-mf-N-T14d",
}

REMOTE_DATASETS = {
    "acousticbrainz_genre": {
        "local_index": "tests/resources/download/acousticbrainz_genre_dataset_little_test.json.zip",
        "filename": "acousticbrainz_genre_dataset_little_test.json",
        "remote_filename": "acousticbrainz_genre_dataset_little_test.json.zip",
        "remote_checksum": "c5fbdd4f8b7de383796a34143cb44c4f",
    }
}


def create_remote_index(httpserver, dataset_name):
    httpserver.serve_content(
        open(REMOTE_DATASETS[dataset_name]["local_index"], "rb").read()
    )
    remote_index = {
        "index": download_utils.RemoteFileMetadata(
            filename=REMOTE_DATASETS[dataset_name]["remote_filename"],
            url=httpserver.url,
            checksum=REMOTE_DATASETS[dataset_name]["remote_checksum"],
            destination_dir="",
        )
    }
    data_remote = core.LargeData(
        REMOTE_DATASETS[dataset_name]["filename"], remote_index=remote_index
    )
    return data_remote.index


def clean_remote_dataset(dataset_name):
    os.remove(
        os.path.join(
            "mirdata/datasets/indexes", REMOTE_DATASETS[dataset_name]["filename"]
        )
    )


def test_dataset_attributes(httpserver):
    for dataset_name in DATASETS:
        module = importlib.import_module("mirdata.datasets.{}".format(dataset_name))
        if dataset_name not in REMOTE_DATASETS:
            dataset = module.Dataset()
        else:
            remote_index = create_remote_index(httpserver, dataset_name)
            dataset = module.Dataset(index=remote_index)

        assert (
            dataset.name == dataset_name
        ), "{}.dataset attribute does not match dataset name".format(dataset_name)
        assert (
            dataset.bibtex is not None
        ), "No BIBTEX information provided for {}".format(dataset_name)
        assert (
                dataset._license_info is not None
        ), "No LICENSE information provided for {}".format(dataset_name)
        assert (
            isinstance(dataset.remotes, dict) or dataset.remotes is None
        ), "{}.REMOTES must be a dictionary".format(dataset_name)
        assert isinstance(dataset._index, dict), "{}.DATA is not properly set".format(
            dataset_name
        )
        assert (
            isinstance(dataset._download_info, str) or dataset._download_info is None
        ), "{}.DOWNLOAD_INFO must be a string".format(dataset_name)
        assert type(dataset._track_object) == type(
            core.Track
        ), "{}.Track must be an instance of core.Track".format(dataset_name)
        assert callable(dataset.download), "{}.download is not a function".format(
            dataset_name
        )

        if dataset_name in REMOTE_DATASETS:
            clean_remote_dataset(dataset_name)


def test_cite_and_license(httpserver):
    for dataset_name in DATASETS:
        module = importlib.import_module("mirdata.datasets.{}".format(dataset_name))
        if dataset_name not in REMOTE_DATASETS:
            dataset = module.Dataset()
        else:
            remote_index = create_remote_index(httpserver, dataset_name)
            dataset = module.Dataset(index=remote_index)

        text_trap = io.StringIO()
        sys.stdout = text_trap
        dataset.cite()
        sys.stdout = sys.__stdout__

        text_trap = io.StringIO()
        sys.stdout = text_trap
        dataset.license()
        sys.stdout = sys.__stdout__
        if dataset_name in REMOTE_DATASETS:
            clean_remote_dataset(dataset_name)


KNOWN_ISSUES = {}  # key is module, value is REMOTE key
DOWNLOAD_EXCEPTIONS = ["maestro", "acousticbrainz_genre"]


def test_download(mocker, httpserver):
    for dataset_name in DATASETS:
        print(dataset_name)
        module = importlib.import_module("mirdata.datasets.{}".format(dataset_name))
        if dataset_name not in REMOTE_DATASETS:
            dataset = module.Dataset()
        else:
            remote_index = create_remote_index(httpserver, dataset_name)
            dataset = module.Dataset(index=remote_index)

        # test parameters & defaults
        assert callable(dataset.download), "{}.download is not callable".format(
            dataset_name
        )
        params = signature(dataset.download).parameters

        expected_params = [
            ("partial_download", None),
            ("force_overwrite", False),
            ("cleanup", False),
        ]
        for exp in expected_params:
            assert exp[0] in params, "{}.download must have {} as a parameter".format(
                dataset_name, exp[0]
            )
            assert (
                params[exp[0]].default == exp[1]
            ), "The default value of {} in {}.download must be {}".format(
                dataset_name, exp[0], exp[1]
            )

        # check that the download method can be called without errors
        if dataset.remotes != {}:
            mock_downloader = mocker.patch.object(dataset, "remotes")
            if dataset_name not in DOWNLOAD_EXCEPTIONS:
                try:
                    dataset.download()
                except:
                    assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

                mocker.resetall()

            # check that links are online
            for key in dataset.remotes:
                # skip this test if it's in known issues
                if dataset_name in KNOWN_ISSUES and key in KNOWN_ISSUES[dataset_name]:
                    continue

                url = dataset.remotes[key].url
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
        if dataset_name in REMOTE_DATASETS:
            clean_remote_dataset(dataset_name)


# This is magically skipped by the the remote fixture `skip_local` in conftest.py
# when tests are run with the --local flag
def test_validate(skip_local, httpserver):
    for dataset_name in DATASETS:
        data_home = os.path.join("tests/resources/mir_datasets", dataset_name)

        module = importlib.import_module("mirdata.datasets.{}".format(dataset_name))
        if dataset_name not in REMOTE_DATASETS:
            dataset = module.Dataset(data_home)
            dataset_default = module.Dataset(data_home=None)
        else:
            remote_index = create_remote_index(httpserver, dataset_name)
            dataset = module.Dataset(data_home, index=remote_index)
            dataset_default = module.Dataset(data_home=None, index=remote_index)

        try:
            dataset.validate()
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        try:
            dataset.validate(verbose=False)
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        try:
            dataset_default.validate(verbose=False)
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])


def test_load_and_trackids(httpserver):
    for dataset_name in DATASETS:
        data_home = os.path.join("tests/resources/mir_datasets", dataset_name)
        module = importlib.import_module("mirdata.datasets.{}".format(dataset_name))
        if dataset_name not in REMOTE_DATASETS:
            dataset = module.Dataset(data_home)
            dataset_default = module.Dataset()
        else:
            continue
            # TODO - fix the dataset object to work with remote index
            # remote_index = create_remote_index(httpserver, dataset_name)
            # dataset = module.Dataset(data_home, index=remote_index)
            # dataset_default = module.Dataset(index=remote_index)

        try:
            track_ids = dataset.track_ids
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])
        assert type(track_ids) is list, "{}.track_ids() should return a list".format(
            dataset_name
        )
        trackid_len = len(track_ids)
        # if the dataset has tracks, test the loaders
        if dataset._track_object is not None:

            try:
                choice_track = dataset.choice_track()
            except:
                assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])
            assert isinstance(
                choice_track, core.Track
            ), "{}.choice_track must return an instance of type core.Track".format(
                dataset_name
            )

            try:
                dataset_data = dataset.load_tracks()
            except:
                assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

            assert (
                type(dataset_data) is dict
            ), "{}.load should return a dictionary".format(dataset_name)
            assert (
                len(dataset_data.keys()) == trackid_len
            ), "the dictionary returned {}.load() does not have the same number of elements as {}.track_ids()".format(
                dataset_name, dataset_name
            )

            try:
                dataset_data_default = dataset_default.load_tracks()
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
        if dataset_name in REMOTE_DATASETS:
            clean_remote_dataset(dataset_name)


def test_track(httpserver):
    data_home_dir = "tests/resources/mir_datasets"

    for dataset_name in DATASETS:
        data_home = os.path.join(data_home_dir, dataset_name)

        module = importlib.import_module("mirdata.datasets.{}".format(dataset_name))
        if dataset_name not in REMOTE_DATASETS:
            dataset = module.Dataset(data_home)
            dataset_default = module.Dataset()
        else:
            continue
            # TODO - fix the dataset object to work with remote index
            # remote_index = create_remote_index(httpserver, dataset_name)
            # dataset = module.Dataset(data_home, index=remote_index)
            # dataset_default = module.Dataset(index=remote_index)

        # if the dataset doesn't have a track object, make sure it raises a value error
        # and move on to the next dataset
        if dataset._track_object is None:
            with pytest.raises(NotImplementedError):
                dataset.track("~faketrackid~?!")
            continue

        if dataset_name in CUSTOM_TEST_TRACKS:
            trackid = CUSTOM_TEST_TRACKS[dataset_name]
        else:
            trackid = dataset.track_ids[0]

        try:
            track_default = dataset_default.track(trackid)
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        assert track_default._data_home == os.path.join(
            DEFAULT_DATA_HOME, dataset.name
        ), "{}: Track._data_home path is not set as expected".format(dataset_name)

        # test data home specified
        try:
            track_test = dataset.track(trackid)
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        assert isinstance(
            track_test, core.Track
        ), "{}.track must be an instance of type core.Track".format(dataset_name)

        assert hasattr(
            track_test, "to_jams"
        ), "{}.track must have a to_jams method".format(dataset_name)

        # Validate JSON schema
        try:
            jam = track_test.to_jams()
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        assert jam.validate(), "Jams validation failed for {}.track({})".format(
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
            dataset.track("~faketrackid~?!")


# for load_* functions which require more than one argument
# module_name : {function_name: {parameter2: value, parameter3: value}}
EXCEPTIONS = {
    "dali": {"load_annotations_granularity": {"granularity": "notes"}},
    "guitarset": {
        "load_pitch_contour": {"string_num": 1},
        "load_notes": {"string_num": 1},
    },
    "saraga": {
        "load_tempo": {"iam_style": "carnatic"},
        "load_sections": {"iam_style": "carnatic"},
    },
}
SKIP = {
    "acousticbrainz_genre": [
        "load_all_train",
        "load_all_validation",
        "load_tagtraum_validation",
        "load_tagtraum_train",
        "load_allmusic_train",
        "load_allmusic_validation",
        "load_lastfm_train",
        "load_lastfm_validation",
        "load_discogs_train",
        "load_discogs_validation",
    ]
}


def test_load_methods(httpserver):
    for dataset_name in DATASETS:
        module = importlib.import_module("mirdata.datasets.{}".format(dataset_name))
        if dataset_name not in REMOTE_DATASETS:
            dataset = module.Dataset()
        else:
            remote_index = create_remote_index(httpserver, dataset_name)
            dataset = module.Dataset(index=remote_index)

        all_methods = dir(dataset)
        load_methods = [
            getattr(dataset, m) for m in all_methods if m.startswith("load_")
        ]
        for load_method in load_methods:
            method_name = load_method.__name__

            # skip default methods
            if method_name == "load_tracks":
                continue

            # skip overrides, add to the SKIP dictionary to skip a specific load method
            if dataset_name in SKIP and method_name in SKIP[dataset_name]:
                continue

            if load_method.__doc__ is None:
                raise ValueError("{} has no documentation".format(method_name))

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


def test_multitracks(httpserver):
    data_home_dir = "tests/resources/mir_datasets"

    for dataset_name in DATASETS:

        module = importlib.import_module("mirdata.datasets.{}".format(dataset_name))
        if dataset_name not in REMOTE_DATASETS:
            dataset = module.Dataset()
        else:
            remote_index = create_remote_index(httpserver, dataset_name)
            dataset = module.Dataset(index=remote_index)

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
        data_home = os.path.join(data_home_dir, dataset_name)
        dataset_specific = mirdata.initialize(dataset_name, data_home=data_home)
        try:
            mtrack_test = dataset_specific.MultiTrack(mtrack_id, data_home=data_home)
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        assert isinstance(
            mtrack_test, core.MultiTrack
        ), "{}.MultiTrack must be an instance of type core.MultiTrack".format(
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

        assert jam.validate(), "Jams validation failed for {}.MultiTrack({})".format(
            dataset_name, mtrack_id
        )
        if dataset_name in REMOTE_DATASETS:
            clean_remote_dataset(dataset_name)
