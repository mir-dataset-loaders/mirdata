from __future__ import absolute_import

import os
import json

import pytest

from mirdata import beatles, utils
from tests.test_utils import (mock_validated, mock_download, mock_untar,
                              mock_validator, mock_clobber_all)


@pytest.fixture
def mock_validate(mocker):
    return mocker.patch.object(beatles, 'validate')


@pytest.fixture
def data_home(tmpdir):
    return str(tmpdir)


@pytest.fixture
def save_path(data_home):
    return utils.get_save_path(data_home)


@pytest.fixture
def dataset_path(save_path):
    return os.path.join(save_path, beatles.BEATLES_DIR)


def test_download_already_valid(data_home, mocker,
                                mock_clobber_all,
                                mock_validated,
                                mock_validator,
                                mock_download,
                                mock_untar):
    mock_validated.return_value = True

    beatles.download(data_home)

    mock_clobber_all.assert_not_called()
    mock_validated.assert_called_once()
    mock_download.assert_not_called()
    mock_untar.assert_not_called()
    mock_validator.assert_not_called()


def test_download_clean(data_home,
                        dataset_path,
                        mocker,
                        mock_clobber_all,
                        mock_validated,
                        mock_download,
                        mock_untar,
                        mock_validate):

    mock_validated.return_value = False
    mock_download.return_value = "foobar"
    mock_untar.return_value = ""
    mock_validate.return_value = (False, False)

    beatles.download(data_home)

    mock_clobber_all.assert_not_called()
    mock_validated.assert_called_once()
    mock_download.assert_called_once()
    mock_untar.assert_called_once_with(mock_download.return_value, dataset_path, cleanup=True)
    mock_validate.assert_called_once_with(dataset_path, data_home)


def test_download_clobber(data_home,
                          dataset_path,
                          mocker,
                          mock_clobber_all,
                          mock_validated,
                          mock_download,
                          mock_untar,
                          mock_validate):

    mock_validated.return_value = False
    mock_download.return_value = "foobar"
    mock_untar.return_value = ""
    mock_validate.return_value = (False, False)

    beatles.download(data_home, clobber=True)

    mock_clobber_all.assert_called_once_with(beatles.BEATLES_ANNOT_REMOTE, dataset_path, data_home)
    mock_validated.assert_called_once()
    mock_download.assert_called_once()
    mock_untar.assert_called_once_with(mock_download.return_value, dataset_path, cleanup=True)
    mock_validate.assert_called_once_with(dataset_path, data_home)


def test_validate_invalid(dataset_path, mocker, mock_validator):
    mock_validator.return_value = (True, True)

    missing_files, invalid_checksums = beatles.validate(dataset_path)
    assert missing_files and invalid_checksums
    mock_validator.assert_called_once()


def test_validate_valid(dataset_path, mocker, mock_validator):
    mock_validator.return_value = (False, False)

    missing_files, invalid_checksums = beatles.validate(dataset_path)
    assert not (missing_files or invalid_checksums)
    mock_validator.assert_called_once()
