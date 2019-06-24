from __future__ import absolute_import

import os

import pytest

from mirdata import medleydb_pitch, utils
from tests.test_utils import mock_validated, mock_validator


@pytest.fixture
def mock_validate(mocker):
    return mocker.patch.object(medleydb_pitch, 'validate')


@pytest.fixture
def data_home(tmpdir):
    return str(tmpdir)


@pytest.fixture
def save_path(data_home):
    return utils.get_save_path(data_home)


@pytest.fixture
def dataset_path(save_path):
    return os.path.join(save_path, medleydb_pitch.DATASET_DIR)


def test_validate_valid(dataset_path, mocker, mock_validator):
    mock_validator.return_value = (False, False)

    missing_files, invalid_checksums = medleydb_pitch.validate(dataset_path)
    assert not (missing_files or invalid_checksums)
    mock_validator.assert_called_once()
