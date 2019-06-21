from __future__ import absolute_import

import os
import json

import pytest

from mirdata import medleydb_melody, utils
from tests.test_utils import mock_validated, mock_validator, mock_force_overwrite_all


@pytest.fixture
def mock_validate(mocker):
    return mocker.patch.object(medleydb_melody, 'validate')


@pytest.fixture
def data_home(tmpdir):
    return str(tmpdir)


@pytest.fixture
def save_path(data_home):
    return utils.get_save_path(data_home)


@pytest.fixture
def dataset_path(save_path):
    return os.path.join(save_path, medleydb_melody.MEDLEYDB_MELODY_DIR)


def test_download_already_valid(data_home, mocker,
                                mock_force_overwrite_all,
                                mock_validated,
                                mock_validator):
    mock_validated.return_value = True

    medleydb_melody.download(data_home)

    mock_force_overwrite_all.assert_not_called()
    mock_validated.assert_called_once()
    mock_validator.assert_not_called()


def test_download_clean(data_home,
                        save_path,
                        dataset_path,
                        mocker,
                        mock_force_overwrite_all,
                        mock_validated,
                        mock_validate):

    mock_validated.return_value = False
    mock_validate.return_value = (False, False)

    medleydb_melody.download(data_home)

    mock_force_overwrite_all.assert_not_called()
    mock_validated.assert_called_once()
    mock_validate.assert_called_once_with(dataset_path, data_home)


def test_download_force_overwrite(data_home,
                          save_path,
                          dataset_path,
                          mocker,
                          mock_force_overwrite_all,
                          mock_validated,
                          mock_validate):

    mock_validated.return_value = False
    mock_validate.return_value = (False, False)

    medleydb_melody.download(data_home, force_overwrite=True)

    mock_force_overwrite_all.assert_called_once_with(medleydb_melody.MEDLEYDB_METADATA, dataset_path, data_home)
    mock_validated.assert_called_once()
    mock_validate.assert_called_once_with(dataset_path, data_home)


def test_validate_invalid(dataset_path, mocker, mock_validator):
    mock_validator.return_value = (True, True)

    missing_files, invalid_checksums = medleydb_melody.validate(dataset_path)
    assert missing_files and invalid_checksums
    mock_validator.assert_called_once()


def test_validate_valid(dataset_path, mocker, mock_validator):
    mock_validator.return_value = (False, False)

    missing_files, invalid_checksums = medleydb_melody.validate(dataset_path)
    assert not (missing_files or invalid_checksums)
    mock_validator.assert_called_once()
