from __future__ import absolute_import

import os

import pytest

from mirdata import orchset, utils
from tests.test_utils import (mock_validated, mock_download, mock_unzip,
                              mock_validator, mock_force_delete_all)


@pytest.fixture
def mock_validate(mocker):
    return mocker.patch.object(orchset, 'validate')


@pytest.fixture
def data_home(tmpdir):
    return str(tmpdir)


@pytest.fixture
def mock_orchset_exists(mocker):
    return mocker.patch.object(os.path, 'exists')


def test_download_already_exists(data_home, mocker,
                                 mock_force_delete_all,
                                 mock_orchset_exists,
                                 mock_download,
                                 mock_unzip):
    mock_orchset_exists.return_value = True

    orchset.download(data_home)

    mock_force_delete_all.assert_not_called()
    mock_orchset_exists.assert_called_once()
    mock_download.assert_not_called()


def test_download_clean(data_home,
                        mocker,
                        mock_force_delete_all,
                        mock_download,
                        mock_unzip,
                        mock_orchset_exists):

    mock_orchset_exists.return_value = False
    mock_download.return_value = 'foobar'
    mock_unzip.return_value = ''

    orchset.download(data_home)

    mock_force_delete_all.assert_not_called()
    mock_orchset_exists.assert_called_once()
    mock_download.assert_called_once()
    mock_unzip.assert_called_once_with(mock_download.return_value, data_home)


def test_download_force_overwrite(data_home,
                          mocker,
                          mock_force_delete_all,
                          mock_orchset_exists,
                          mock_download,
                          mock_unzip):

    mock_orchset_exists.return_value = False
    mock_download.return_value = 'foobar'
    mock_unzip.return_value = ''

    orchset.download(data_home, force_overwrite=True)

    mock_force_delete_all.assert_called_once_with(orchset.REMOTE, data_home)
    mock_orchset_exists.assert_called_once()
    mock_download.assert_called_once()
    mock_unzip.assert_called_once_with(mock_download.return_value, data_home)


def test_validate_invalid(data_home, mocker, mock_validator):
    mock_validator.return_value = (True, True)

    missing_files, invalid_checksums = orchset.validate(data_home)
    assert missing_files and invalid_checksums
    mock_validator.assert_called_once()


def test_validate_valid(data_home, mocker, mock_validator):
    mock_validator.return_value = (False, False)

    missing_files, invalid_checksums = orchset.validate(data_home)
    assert not (missing_files or invalid_checksums)
    mock_validator.assert_called_once()
