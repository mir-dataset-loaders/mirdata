from __future__ import absolute_import

import os
import sys

from mirdata import utils

import json
import pytest

if sys.version_info.major == 3:
    builtin_module_name = 'builtins'
else:
    builtin_module_name = '__builtin__'

DEFAULT_DATA_HOME = os.path.join(os.getenv('HOME', '/tmp'), 'mir_datasets')


@pytest.fixture
def mock_validated(mocker):
    return mocker.patch.object(utils, 'check_validated')


@pytest.fixture
def mock_validator(mocker):
    return mocker.patch.object(utils, 'validator')


@pytest.fixture
def mock_check_index(mocker):
    return mocker.patch.object(utils, 'check_index')


def test_md5(mocker):
    audio_file = b'audio1234'

    expected_checksum = '6dc00d1bac757abe4ea83308dde68aab'

    mocker.patch(
        '%s.open' % builtin_module_name, new=mocker.mock_open(read_data=audio_file)
    )

    md5_checksum = utils.md5('test_file_path')
    assert expected_checksum == md5_checksum


@pytest.mark.parametrize(
    'test_index,expected_missing,expected_inv_checksum',
    [
        ('test_index_valid.json', {}, {}),
        (
            'test_index_missing_file.json',
            {'10161_chorus': ['tests/resources/10162_chorus.wav']},
            {},
        ),
        (
            'test_index_invalid_checksum.json',
            {},
            {'10161_chorus': ['tests/resources/10161_chorus.wav']},
        ),
    ],
)
def test_check_index(test_index, expected_missing, expected_inv_checksum):
    index_path = os.path.join('tests/indexes', test_index)
    with open(index_path) as index_file:
        test_index = json.load(index_file)

    missing_files, invalid_checksums = utils.check_index(
        test_index, 'tests/resources/', True
    )

    assert expected_missing == missing_files
    assert expected_inv_checksum == invalid_checksums


@pytest.mark.parametrize(
    'missing_files,invalid_checksums',
    [
        ({'10161_chorus': ['tests/resources/10162_chorus.wav']}, {}),
        ({}, {'10161_chorus': ['tests/resources/10161_chorus.wav']}),
    ],
)
def test_validator(mocker, mock_check_index, missing_files, invalid_checksums):
    mock_check_index.return_value = missing_files, invalid_checksums

    m, c = utils.validator('foo', 'bar', True)
    assert m == missing_files
    assert c == invalid_checksums
    mock_check_index.assert_called_once_with('foo', 'bar', True)

    # if missing_files or invalid_checksums:
    #     mock_create_invalid.assert_called_once_with(missing_files, invalid_checksums)
    # else:
    #     mock_create_validated.assert_called_once_with('baz')


def test_get_default_dataset_path():
    assert '/tmp/mir_datasets/data_home' == utils.get_default_dataset_path('data_home')
