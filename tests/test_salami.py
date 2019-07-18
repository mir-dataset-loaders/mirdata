from __future__ import absolute_import

import os
import pytest
from mirdata import salami, utils
from tests.test_utils import (mock_download, mock_unzip,
                              mock_validator, mock_force_delete_all)


@pytest.fixture
def mock_validate(mocker):
    return mocker.patch.object(salami, 'validate')


@pytest.fixture
def mock_load_sections(mocker):
    return mocker.patch.object(salami, '_load_sections')


@pytest.fixture
def data_home(tmpdir):
    return str(tmpdir)


@pytest.fixture
def mock_salami_exists(mocker):
    return mocker.patch.object(os.path, 'exists')


def test_download_already_exists(data_home, mocker,
                                 mock_force_delete_all,
                                 mock_salami_exists,
                                 mock_validator,
                                 mock_download,
                                 mock_unzip):
    mock_salami_exists.return_value = True

    salami.download(data_home)

    mock_force_delete_all.assert_not_called()
    mock_salami_exists.assert_called_once()
    mock_download.assert_not_called()
    mock_unzip.assert_not_called()
    mock_validator.assert_not_called()


def test_download_clean(data_home,
                        mocker,
                        mock_force_delete_all,
                        mock_salami_exists,
                        mock_download,
                        mock_unzip,
                        mock_validate):

    mock_salami_exists.return_value = False
    mock_download.return_value = 'foobar'
    mock_unzip.return_value = ''
    mock_validate.return_value = (False, False)

    salami.download(data_home)

    mock_force_delete_all.assert_not_called()
    mock_salami_exists.assert_called_once()
    mock_download.assert_called_once()
    mock_unzip.assert_called_once_with(mock_download.return_value, data_home, cleanup=True)
    mock_validate.assert_called_once_with(data_home)


def test_download_force_overwrite(data_home,
                          mocker,
                          mock_force_delete_all,
                          mock_salami_exists,
                          mock_download,
                          mock_unzip,
                          mock_validate):

    mock_salami_exists.return_value = False
    mock_download.return_value = 'foobar'
    mock_unzip.return_value = ''
    mock_validate.return_value = (False, False)

    salami.download(data_home, force_overwrite=True)

    mock_force_delete_all.assert_called_once_with(salami.ANNOTATIONS_REMOTE, data_home=data_home)
    mock_salami_exists.assert_called_once()
    mock_download.assert_called_once()
    mock_unzip.assert_called_once_with(mock_download.return_value, data_home, cleanup=True)
    mock_validate.assert_called_once_with(data_home)


def test_validate_invalid(data_home, mocker, mock_validator):
    mock_validator.return_value = (True, True)

    missing_files, invalid_checksums = salami.validate(data_home)
    assert missing_files and invalid_checksums
    mock_validator.assert_called_once()


def test_validate_valid(data_home, mocker, mock_validator):
    mock_validator.return_value = (False, False)

    missing_files, invalid_checksums = salami.validate(data_home)
    assert not (missing_files or invalid_checksums)
    mock_validator.assert_called_once()


def test_load_track_invalid_track_id():
    with pytest.raises(ValueError):
        salami.Track('a_fake_track')


def test_load_track_no_metadata():
    with pytest.raises(OSError):
        salami.METADATA = None
        salami.Track('10')


def test_load_track_wrong_metadata():
    with pytest.raises(OSError):
        salami.METADATA = {'data_home': 'test'}
        salami.Track('10', 'wrong_data_home')


# def test_load_track_missing_metadata(mock_load_sections):
#     data_home = 'fake-data-home'
#     salami.METADATA = {'data_home': data_home}
#     mock_load_sections.return_value = ['a', 'b', 'c', 'd']

#     expected_track = salami.SalamiTrack(
#         '10',
#         'fake-data-home/Salami/audio/10.mp3',
#         'a',
#         'b',
#         'c',
#         'd',
#         None,
#         None,
#         None,
#         None,
#         None,
#         None,
#         None,
#         None,
#         None,
#         None,
#     )

#     actual_track = salami.load_track('10', data_home)
#     assert actual_track == expected_track


# def test_load_track_with_metadata(mock_load_sections):
#     track_metadata = {
#         'source': 'source', 'annotator_1_id': 'aid-1', 'annotator_2_id': 'aid-2',
#         'duration_sec': 60, 'title': 'title', 'artist': 'artist',
#         'annotator_1_time': None, 'annotator_2_time': None, 'class': 'class',
#         'genre': 'genre'
#     }
#     data_home = 'fake-data-home'
#     salami.METADATA = {'data_home': data_home, '10': track_metadata}
#     mock_load_sections.return_value = ['a', 'b', 'c', 'd']

#     expected_track = salami.SalamiTrack(
#         '10',
#         'fake-data-home/Salami/audio/10.mp3',
#         'a',
#         'b',
#         'c',
#         'd',
#         'source',
#         'aid-1',
#         'aid-2',
#         60,
#         'title',
#         'artist',
#         None,
#         None,
#         'class',
#         'genre',
#     )

#     actual_track = salami.load_track('10', data_home)
#     assert actual_track == expected_track


def test_track_ids():
    assert salami.track_ids() == list(salami.INDEX.keys())
