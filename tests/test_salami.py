from __future__ import absolute_import

import os
import pytest
try:
    from mock import MagicMock  # py2
except ImportError:
    from unittest.mock import MagicMock  # py3
from mirdata import salami, utils
from tests.test_utils import (mock_validated, mock_download, mock_unzip,
                              mock_validator, mock_clobber_all)


@pytest.fixture
def mock_validate(mocker):
    return mocker.patch.object(salami, 'validate')


@pytest.fixture
def data_home(tmpdir):
    return str(tmpdir)


@pytest.fixture
def save_path(data_home):
    return utils.get_save_path(data_home)


@pytest.fixture
def dataset_path(save_path):
    return os.path.join(save_path, salami.SALAMI_DIR)


def test_download_already_valid(data_home, mocker,
                                mock_clobber_all,
                                mock_validated,
                                mock_validator,
                                mock_download,
                                mock_unzip):
    mock_validated.return_value = True

    salami.download(data_home)

    mock_clobber_all.assert_not_called()
    mock_validated.assert_called_once()
    mock_download.assert_not_called()
    mock_unzip.assert_not_called()
    mock_validator.assert_not_called()


def test_download_clean(data_home,
                        dataset_path,
                        mocker,
                        mock_clobber_all,
                        mock_validated,
                        mock_download,
                        mock_unzip,
                        mock_validate):

    mock_validated.return_value = False
    mock_download.return_value = "foobar"
    mock_unzip.return_value = ""
    mock_validate.return_value = (False, False)

    salami.download(data_home)

    mock_clobber_all.assert_not_called()
    mock_validated.assert_called_once()
    mock_download.assert_called_once()
    mock_unzip.assert_called_once_with(mock_download.return_value, dataset_path, cleanup=True)
    mock_validate.assert_called_once_with(dataset_path, data_home)


def test_download_clobber(data_home,
                          dataset_path,
                          mocker,
                          mock_clobber_all,
                          mock_validated,
                          mock_download,
                          mock_unzip,
                          mock_validate):

    mock_validated.return_value = False
    mock_download.return_value = "foobar"
    mock_unzip.return_value = ""
    mock_validate.return_value = (False, False)

    salami.download(data_home, clobber=True)

    mock_clobber_all.assert_called_once_with(salami.SALAMI_ANNOT_REMOTE, dataset_path, data_home)
    mock_validated.assert_called_once()
    mock_download.assert_called_once()
    mock_unzip.assert_called_once_with(mock_download.return_value, dataset_path, cleanup=True)
    mock_validate.assert_called_once_with(dataset_path, data_home)


def test_validate_invalid(dataset_path, mocker, mock_validator):
    mock_validator.return_value = (True, True)

    missing_files, invalid_checksums = salami.validate(dataset_path)
    assert missing_files and invalid_checksums
    mock_validator.assert_called_once()


def test_validate_valid(dataset_path, mocker, mock_validator):
    mock_validator.return_value = (False, False)

    missing_files, invalid_checksums = salami.validate(dataset_path)
    assert not (missing_files or invalid_checksums)
    mock_validator.assert_called_once()


def test_load_track_invalid_track_id():
    with pytest.raises(ValueError):
        salami.load_track('a_fake_track')


def test_load_track_no_metadata():
    with pytest.raises(EnvironmentError):
        salami.SALAMI_METADATA = None
        salami.load_track('10')


def test_load_track_wrong_metadata():
    with pytest.raises(EnvironmentError):
        salami.SALAMI_METADATA = {'data_home': 'test'}
        salami.load_track('10', 'wrong_data_home')


def test_load_track_missing_metadata():
    data_home = 'fake-data-home'
    salami.SALAMI_METADATA = {'data_home': data_home}
    salami.load_sections = MagicMock(return_value=['a', 'b', 'c', 'd'])

    expected_track = salami.SalamiTrack(
        '10',
        'fake-data-home/Salami/audio/10.mp3',
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )

    actual_track = salami.load_track('10', data_home)
    assert actual_track == expected_track


def test_track_ids():
    assert salami.track_ids() == list(salami.SALAMI_INDEX.keys())
