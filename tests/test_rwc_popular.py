from __future__ import absolute_import

import os

import numpy as np
import pytest

from mirdata import rwc_popular, utils
from tests.test_utils import (mock_validated, mock_download, mock_untar,
                              mock_validator, mock_force_delete_all)


@pytest.fixture
def mock_validate(mocker):
    return mocker.patch.object(rwc_popular, 'validate')


@pytest.fixture
def data_home(tmpdir):
    return str(tmpdir)


@pytest.fixture
def save_path(data_home):
    return utils.get_save_path(data_home)


@pytest.fixture
def dataset_path(save_path):
    return os.path.join(save_path, rwc_popular.DATASET_DIR)


@pytest.fixture
def mock_rwc_popular_exists(mocker):
    return mocker.patch.object(rwc_popular, 'exists')


def test_download_already_exists(data_home, mocker,
                                 mock_force_delete_all,
                                 mock_rwc_popular_exists,
                                 mock_validator,
                                 mock_download,
                                 mock_untar):
    mock_rwc_popular_exists.return_value = True

    rwc_popular.download(data_home)

    mock_force_delete_all.assert_not_called()
    mock_download.assert_not_called()
    mock_untar.assert_not_called()
    mock_validator.assert_not_called()


def test_download_clean(data_home,
                        dataset_path,
                        mocker,
                        mock_force_delete_all,
                        mock_rwc_popular_exists,
                        mock_download,
                        mock_untar,
                        mock_validate):
    mock_rwc_popular_exists.return_value = False
    mock_download.return_value = 'foobar'
    mock_untar.return_value = ''
    mock_validate.return_value = (False, False)

    rwc_popular.download(data_home)

    mock_force_delete_all.assert_not_called()
    mock_rwc_popular_exists.assert_called_once()
    mock_download.assert_called_once()
    mock_untar.assert_called_once_with(mock_download.return_value, dataset_path, cleanup=True)
    mock_validate.assert_called_once_with(dataset_path, data_home)


def test_download_force_overwrite(data_home,
                                  dataset_path,
                                  mocker,
                                  mock_force_delete_all,
                                  mock_beatles_exists,
                                  mock_download,
                                  mock_untar,
                                  mock_validate):

    mock_beatles_exists.return_value = False
    mock_download.return_value = 'foobar'
    mock_untar.return_value = ''
    mock_validate.return_value = (False, False)

    rwc_popular.download(data_home, force_overwrite=True)

    mock_force_delete_all.assert_called_once_with(rwc_popular.ANNOTATIONS_REMOTE_1,
                                                  dataset_path=None, data_home=data_home)
    mock_force_delete_all.assert_called_once_with(rwc_popular.ANNOTATIONS_REMOTE_2,
                                                  dataset_path=None, data_home=data_home)
    mock_force_delete_all.assert_called_once_with(rwc_popular.ANNOTATIONS_REMOTE_3,
                                                  dataset_path=None, data_home=data_home)
    mock_force_delete_all.assert_called_once_with(rwc_popular.ANNOTATIONS_REMOTE_4,
                                                  dataset_path=None, data_home=data_home)
    mock_beatles_exists.assert_called_once()
    mock_download.assert_called_once()
    mock_untar.assert_called_once_with(mock_download.return_value, dataset_path, cleanup=True)
    mock_validate.assert_called_once_with(dataset_path, data_home)


def test_validate_invalid(dataset_path, mocker, mock_validator):
    mock_validator.return_value = (True, True)

    missing_files, invalid_checksums = rwc_popular.validate(dataset_path)
    assert missing_files and invalid_checksums
    mock_validator.assert_called_once()


def test_validate_valid(dataset_path, mocker, mock_validator):
    mock_validator.return_value = (False, False)

    missing_files, invalid_checksums = rwc_popular.validate(dataset_path)
    assert not (missing_files or invalid_checksums)
    mock_validator.assert_called_once()


def test_track_ids():
    assert rwc_popular.track_ids() == list(rwc_popular.INDEX.keys())


def test_load_track_invalid_track_id():
    with pytest.raises(ValueError):
        rwc_popular.Track('a-fake-track-id')
