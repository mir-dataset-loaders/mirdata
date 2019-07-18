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


@pytest.fixture
def mock_validated(mocker):
    return mocker.patch.object(utils, 'check_validated')


@pytest.fixture
def mock_download(mocker):
    return mocker.patch.object(utils, 'download_from_remote')


@pytest.fixture
def mock_untar(mocker):
    return mocker.patch.object(utils, 'untar')


@pytest.fixture
def mock_unzip(mocker):
    return mocker.patch.object(utils, 'unzip')


@pytest.fixture
def mock_validator(mocker):
    return mocker.patch.object(utils, 'validator')


@pytest.fixture
def mock_force_delete_all(mocker):
    return mocker.patch.object(utils, 'force_delete_all')


@pytest.fixture
def mock_check_index(mocker):
    return mocker.patch.object(utils, 'check_index')


@pytest.fixture
def mock_create_invalid(mocker):
    return mocker.patch.object(utils, 'create_invalid')


@pytest.fixture
def mock_create_validated(mocker):
    return mocker.patch.object(utils, 'create_validated')


def test_md5(mocker):
    audio_file = b'audio1234'

    expected_checksum = '6dc00d1bac757abe4ea83308dde68aab'

    mocker.patch('%s.open' % builtin_module_name, new=mocker.mock_open(read_data=audio_file))

    md5_checksum = utils.md5('test_file_path')
    assert expected_checksum == md5_checksum


@pytest.mark.parametrize('test_index,expected_missing,expected_inv_checksum', [
    ('test_index_valid.json', {}, {}),
    ('test_index_missing_file.json', {'10161_chorus': ['tests/resources/10162_chorus.wav']}, {}),
    ('test_index_invalid_checksum.json', {}, {'10161_chorus': ['tests/resources/10161_chorus.wav']}),
])
def test_check_index(test_index,
                     expected_missing,
                     expected_inv_checksum):
    index_path = os.path.join('tests/indexes', test_index)
    with open(index_path) as index_file:
        test_index = json.load(index_file)

    missing_files, invalid_checksums = utils.check_index(test_index, 'tests/resources/', True)

    assert expected_missing == missing_files
    assert expected_inv_checksum == invalid_checksums


@pytest.mark.parametrize('missing_files,invalid_checksums', [
    ({'10161_chorus': ['tests/resources/10162_chorus.wav']}, {}),
    ({}, {'10161_chorus': ['tests/resources/10161_chorus.wav']}),
])
def test_validator(mocker,
                   mock_check_index,
                   missing_files,
                   invalid_checksums):
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


def test_download_from_remote(httpserver, tmpdir):
    httpserver.serve_content(open('tests/resources/remote.wav').read())

    TEST_REMOTE = utils.RemoteFileMetadata(
        filename='remote.wav',
        url=httpserver.url,
        checksum=('3f77d0d69dc41b3696f074ad6bf2852f')
    )

    download_path = utils.download_from_remote(TEST_REMOTE, str(tmpdir))
    expected_download_path = os.path.join(str(tmpdir), 'remote.wav')
    assert expected_download_path == download_path


def test_download_from_remote_raises_IOError(httpserver, tmpdir):
    httpserver.serve_content('File not found!', 404)

    TEST_REMOTE = utils.RemoteFileMetadata(
        filename='remote.wav',
        url=httpserver.url,
        checksum=('1234')
    )

    with pytest.raises(IOError):
        utils.download_from_remote(TEST_REMOTE, str(tmpdir))


def test_unzip(tmpdir):
    utils.unzip('tests/resources/remote.zip', str(tmpdir))

    expected_file_location = os.path.join(str(tmpdir), 'remote.wav')
    assert os.path.exists(expected_file_location)


def test_untar(tmpdir):
    utils.untar('tests/resources/remote.tar.gz', str(tmpdir))

    expected_file_location = os.path.join(str(tmpdir), 'remote.wav')
    assert os.path.exists(expected_file_location)


# def test_check_validated(tmpdir):
#     tmpdir_str = str(tmpdir)
#     assert not utils.check_validated(tmpdir_str)

#     utils.create_validated(tmpdir_str)
#     assert utils.check_validated(tmpdir_str)


# def test_create_validated(tmpdir):
#     tmpdir_str = str(tmpdir)
#     expected_validated_path = os.path.join(tmpdir_str, utils.VALIDATED_FILE_NAME)
#     assert not os.path.exists(expected_validated_path)

#     utils.create_validated(tmpdir_str)
#     assert os.path.exists(expected_validated_path)
#     with open(expected_validated_path, 'r') as f:
#         # Yes we could do not f.read(), but the intentions here are clearer
#         assert f.read() == ''


# @pytest.mark.parametrize('missing_files,invalid_checksums', [
#     ({'10161_chorus': ['tests/resources/10162_chorus.wav']}, {}),
#     ({}, {'10161_chorus': ['tests/resources/10161_chorus.wav']}),
# ])
# def test_create_invalid(tmpdir,
#                         missing_files,
#                         invalid_checksums):
#     tmpdir_str = str(tmpdir)
#     utils.create_invalid(tmpdir_str, missing_files, invalid_checksums)

#     with open(os.path.join(tmpdir_str, utils.INVALID_FILE_NAME)) as f:
#         invalid_content = json.load(f)
#         assert invalid_content == {'missing_files': missing_files,
#                                    'invalid_checksums': invalid_checksums}


def test_force_delete_all_nonempty_data_home(httpserver, tmpdir):
    tmpdir_str = str(tmpdir)
    remote_filename = 'remote.wav'
    TEST_REMOTE = utils.RemoteFileMetadata(
        filename=remote_filename,
        url=httpserver.url,
        checksum=('1234')
    )

    with pytest.raises(IOError):
        utils.download_from_remote(TEST_REMOTE, tmpdir_str)

    utils.untar('tests/resources/remote.tar.gz', tmpdir_str)
    assert os.path.exists(os.path.join(tmpdir_str, remote_filename))
    assert os.path.exists(tmpdir_str)
    utils.force_delete_all(TEST_REMOTE, tmpdir_str)
    assert not os.path.exists(os.path.join(tmpdir_str, remote_filename))
    assert not os.path.exists(tmpdir_str)
