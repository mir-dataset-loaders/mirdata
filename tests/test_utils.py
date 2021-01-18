# -*- coding: utf-8 -*-

import itertools
import os
import sys
import types

import mirdata
from mirdata import core, validate, download_utils

import json
import pytest

from mirdata.core import LargeData

if sys.version_info.major == 3:
    builtin_module_name = "builtins"
else:
    builtin_module_name = "__builtin__"

DEFAULT_DATA_HOME = os.path.join(os.getenv("HOME", "/tmp"), "mir_datasets")


def run_track_tests(track, expected_attributes, expected_property_types):
    track_attr = get_attributes_and_properties(track)

    # test track attributes
    for attr in track_attr["attributes"]:
        print("{}: {}".format(attr, getattr(track, attr)))
        assert expected_attributes[attr] == getattr(track, attr)

    # test track property types
    for prop in track_attr["cached_properties"]:
        print("{}: {}".format(prop, type(getattr(track, prop))))
        assert isinstance(getattr(track, prop), expected_property_types[prop])


def get_attributes_and_properties(class_instance):
    attributes = []
    properties = []
    cached_properties = []
    functions = []
    for val in dir(class_instance.__class__):
        if val.startswith("_"):
            continue

        attr = getattr(class_instance.__class__, val)
        if isinstance(attr, mirdata.core.cached_property):
            cached_properties.append(val)
        elif isinstance(attr, property):
            properties.append(val)
        elif isinstance(attr, types.FunctionType):
            functions.append(val)
        else:
            raise ValueError("Unknown type {}".format(attr))

    non_attributes = list(
        itertools.chain.from_iterable([properties, cached_properties, functions])
    )
    for val in dir(class_instance):
        if val.startswith("_"):
            continue
        if val not in non_attributes:
            attributes.append(val)
    return {
        "attributes": attributes,
        "properties": properties,
        "cached_properties": cached_properties,
        "functions": functions,
    }


@pytest.fixture
def mock_validated(mocker):
    return mocker.patch.object(validate, "check_validated")


@pytest.fixture
def mock_validator(mocker):
    return mocker.patch.object(validate, "validator")


@pytest.fixture
def mock_validate_index(mocker):
    return mocker.patch.object(validate, "validate_index")


def test_remote_index(httpserver):
    httpserver.serve_content(
        open(
            "tests/resources/download/acousticbrainz_genre_dataset_little_test.json",
            "rb",
        ).read()
    )
    REMOTE_INDEX = {
        "remote_index": download_utils.RemoteFileMetadata(
            filename="acousticbrainz_genre_dataset_little_test.json",
            url=httpserver.url,
            checksum="50cf34e2e40e3df4c1cd582d08fa4506",  # the md5 checksum
            destination_dir=".",  # relative path for where to unzip the data, or None
        )
    }

    DATA = LargeData(
        "acousticbrainz_genre_dataset_little_test.json", remote_index=REMOTE_INDEX
    )
    ind = DATA.index
    assert len(ind["tracks"]) == 16
    os.remove("mirdata/datasets/indexes/acousticbrainz_genre_dataset_little_test.json")


def test_md5(mocker):
    audio_file = b"audio1234"

    expected_checksum = "6dc00d1bac757abe4ea83308dde68aab"

    mocker.patch(
        "%s.open" % builtin_module_name, new=mocker.mock_open(read_data=audio_file)
    )

    md5_checksum = validate.md5("test_file_path")
    assert expected_checksum == md5_checksum


@pytest.mark.parametrize(
    "test_index,expected_missing,expected_inv_checksum",
    [
        ("test_index_valid.json", {"tracks": {}}, {"tracks": {}}),
        (
            "test_index_missing_file.json",
            {"tracks": {"10161_chorus": ["tests/resources/10162_chorus.wav"]}},
            {"tracks": {}},
        ),
        (
            "test_index_invalid_checksum.json",
            {"tracks": {}},
            {"tracks": {"10161_chorus": ["tests/resources/10161_chorus.wav"]}},
        ),
    ],
)
def test_validate_index(test_index, expected_missing, expected_inv_checksum):
    index_path = os.path.join("tests/indexes", test_index)
    with open(index_path) as index_file:
        test_index = json.load(index_file)

    missing_files, invalid_checksums = validate.validate_index(
        test_index, "tests/resources/"
    )

    assert expected_missing == missing_files
    assert expected_inv_checksum == invalid_checksums


@pytest.mark.parametrize(
    "missing_files,invalid_checksums",
    [
        (
            {"tracks": {"10161_chorus": ["tests/resources/10162_chorus.wav"]}},
            {"tracks": {}},
        ),
        (
            {"tracks": {}},
            {"tracks": {"10161_chorus": ["tests/resources/10161_chorus.wav"]}},
        ),
        ({"tracks": {}}, {"tracks": {}}),
    ],
)
def test_validator(mocker, mock_validate_index, missing_files, invalid_checksums):
    mock_validate_index.return_value = missing_files, invalid_checksums

    m, c = validate.validator("foo", "bar", False)
    assert m == missing_files
    assert c == invalid_checksums
    mock_validate_index.assert_called_once_with("foo", "bar", False)
