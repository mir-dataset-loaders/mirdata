# -*- coding: utf-8 -*-
"""
This test takes a long time, but it makes sure that the datset can be locally downloaded,
validated successfully, and loaded.
"""
import os
import pytest
import tqdm

from tests.test_utils import get_attributes_and_properties
import mirdata


@pytest.fixture()
def dataset(test_dataset):
    if test_dataset == "":
        return None
    elif test_dataset not in mirdata.DATASETS:
        raise ValueError("{} is not a dataset in mirdata".format(test_dataset))
    data_home = os.path.join("tests/resources/mir_datasets_full", test_dataset)
    return mirdata.Dataset(test_dataset, data_home)


# This is magically skipped by the the remote fixture `skip_remote` in conftest.py
# when tests are run without the --local flag
def test_download(skip_remote, dataset, skip_download):
    if dataset is None:
        pytest.skip()

    # download the dataset
    if not skip_download:
        dataset.download()

        print(
            "If this dataset does not have openly downloadable data, "
            + "follow the instructions printed by the download message and "
            + "rerun this test."
        )


def test_validation(skip_remote, dataset):
    if dataset is None:
        pytest.skip()

    # run validation
    missing_files, invalid_checksums = dataset.validate(verbose=True)

    assert missing_files == {}
    assert invalid_checksums == {}


def test_load(skip_remote, dataset):
    if dataset is None:
        pytest.skip()

    # run load
    all_data = dataset.load_tracks()

    assert isinstance(all_data, dict)

    track_ids = dataset.track_ids
    assert set(track_ids) == set(all_data.keys())

    # test that all attributes and properties can be called
    for track_id in tqdm.tqdm(track_ids):
        track = all_data[track_id]
        track_data = get_attributes_and_properties(track)

        for attr in track_data["attributes"]:
            ret = getattr(track, attr)

        for prop in track_data["properties"]:
            ret = getattr(track, prop)

        for cprop in track_data["cached_properties"]:
            ret = getattr(track, cprop)

        jam = track.to_jams()
        assert jam.validate()
