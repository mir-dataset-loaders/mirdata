# -*- coding: utf-8 -*-

import os
import shutil
import sys

from mirdata import download_utils

import pytest

if sys.version_info.major == 3:
    builtin_module_name = 'builtins'
    from pathlib import Path
else:
    builtin_module_name = '__builtin__'
    from pathlib2 import Path


@pytest.fixture
def mock_file(mocker):
    return mocker.patch.object(download_utils, 'download_from_remote')


@pytest.fixture
def mock_downloader(mocker):
    return mocker.patch.object(download_utils, 'downloader')


@pytest.fixture
def mock_untar(mocker):
    return mocker.patch.object(download_utils, 'untar')


@pytest.fixture
def mock_unzip(mocker):
    return mocker.patch.object(download_utils, 'unzip')


@pytest.fixture
def mock_path(mocker, mock_file):
    return mocker.patch.object(Path, 'mkdir')


def test_downloader(mocker, mock_path, capsys):
    mock_zip = mocker.patch.object(download_utils, 'download_zip_file')
    mock_tar = mocker.patch.object(download_utils, 'download_tar_file')
    mock_file = mocker.patch.object(download_utils, 'download_from_remote')

    zip_remote = download_utils.RemoteFileMetadata(
        filename='remote.zip', url='a', checksum=('1234'), destination_dir=None
    )
    tar_remote = download_utils.RemoteFileMetadata(
        filename='remote.tar.gz', url='a', checksum=('1234'), destination_dir=None
    )

    file_remote = download_utils.RemoteFileMetadata(
        filename='remote.txt', url='a', checksum=('1234'), destination_dir=None
    )

    # Zip only
    download_utils.downloader('a', remotes={'b': zip_remote})
    mock_zip.assert_called_once_with(zip_remote, 'a', False, True)
    mocker.resetall()

    # tar only
    download_utils.downloader('a', remotes={'b': tar_remote})
    mock_tar.assert_called_once_with(tar_remote, 'a', False, True)
    mocker.resetall()

    # file only
    download_utils.downloader('a', remotes={'b': file_remote})
    mock_file.assert_called_once_with(file_remote, 'a', False)
    mocker.resetall()

    # zip and tar
    download_utils.downloader('a', remotes={'b': zip_remote, 'c': tar_remote})
    mock_zip.assert_called_once_with(zip_remote, 'a', False, True)
    mock_tar.assert_called_once_with(tar_remote, 'a', False, True)
    mocker.resetall()

    # zip and file
    download_utils.downloader('a', remotes={'b': zip_remote, 'c': file_remote})
    mock_zip.assert_called_once_with(zip_remote, 'a', False, True)
    mock_file.assert_called_once_with(file_remote, 'a', False)
    mocker.resetall()

    # tar and file
    download_utils.downloader('a', remotes={'b': tar_remote, 'c': file_remote})
    mock_tar.assert_called_once_with(tar_remote, 'a', False, True)
    mock_file.assert_called_once_with(file_remote, 'a', False)
    mocker.resetall()

    # zip and tar and file
    download_utils.downloader(
        'a', remotes={'b': zip_remote, 'c': tar_remote, 'd': file_remote}
    )
    mock_zip.assert_called_once_with(zip_remote, 'a', False, True)
    mock_file.assert_called_once_with(file_remote, 'a', False)
    mock_tar.assert_called_once_with(tar_remote, 'a', False, True)
    mocker.resetall()

    # test partial download
    download_utils.downloader(
        'a',
        remotes={'b': zip_remote, 'c': tar_remote, 'd': file_remote},
        partial_download=['b', 'd'],
    )
    mock_zip.assert_called_once_with(zip_remote, 'a', False, True)
    mock_file.assert_called_once_with(file_remote, 'a', False)
    mocker.resetall()

    # test bad type partial download
    with pytest.raises(ValueError):
        download_utils.downloader(
            'a',
            remotes={'b': zip_remote, 'c': tar_remote, 'd': file_remote},
            partial_download='b',
        )

    with pytest.raises(ValueError):
        download_utils.downloader(
            'a',
            remotes={'b': zip_remote, 'c': tar_remote, 'd': file_remote},
            partial_download=['d', 'e'],
        )

    # test info message
    captured = capsys.readouterr()  # skip everything printed before this
    download_utils.downloader('a', info_message='I am a message!')
    captured = capsys.readouterr()
    assert captured.out == "I am a message!\n"


def test_download_from_remote(httpserver, tmpdir):
    httpserver.serve_content(open('tests/resources/remote.wav').read())

    TEST_REMOTE = download_utils.RemoteFileMetadata(
        filename='remote.wav',
        url=httpserver.url,
        checksum=('3f77d0d69dc41b3696f074ad6bf2852f'),
        destination_dir=None,
    )

    download_path = download_utils.download_from_remote(TEST_REMOTE, str(tmpdir))
    expected_download_path = os.path.join(str(tmpdir), 'remote.wav')
    assert expected_download_path == download_path


def test_download_from_remote_destdir(httpserver, tmpdir):
    httpserver.serve_content(open('tests/resources/remote.wav').read())

    TEST_REMOTE = download_utils.RemoteFileMetadata(
        filename='remote.wav',
        url=httpserver.url,
        checksum=('3f77d0d69dc41b3696f074ad6bf2852f'),
        destination_dir='subfolder',
    )

    download_path = download_utils.download_from_remote(TEST_REMOTE, str(tmpdir))
    expected_download_path = os.path.join(str(tmpdir), 'subfolder', 'remote.wav')
    assert expected_download_path == download_path


def test_download_from_remote_raises_IOError(httpserver, tmpdir):
    httpserver.serve_content('File not found!', 404)

    TEST_REMOTE = download_utils.RemoteFileMetadata(
        filename='remote.wav',
        url=httpserver.url,
        checksum=('1234'),
        destination_dir=None,
    )

    with pytest.raises(IOError):
        download_utils.download_from_remote(TEST_REMOTE, str(tmpdir))


def test_unzip():
    download_utils.unzip('tests/resources/file.zip', cleanup=False)
    expected_file_location = os.path.join('tests', 'resources', 'file.txt')
    assert os.path.exists(expected_file_location)
    os.remove(expected_file_location)


def test_untar():
    download_utils.untar('tests/resources/file.tar.gz', cleanup=False)
    expected_file_location = os.path.join('tests', 'resources', 'file', 'file.txt')
    assert os.path.exists(expected_file_location)
    os.remove(expected_file_location)


def test_download_zip_file(mocker, mock_file, mock_unzip):
    mock_file.return_value = "foo"
    download_utils.download_zip_file("a", "b", True)

    mock_file.assert_called_once_with("a", "b", True)
    mock_unzip.assert_called_once_with("foo", cleanup=True)
    if os.path.exists('a'):
        shutil.rmtree('a')


def test_download_tar_file(mocker, mock_file, mock_untar):
    mock_file.return_value = "foo"
    download_utils.download_tar_file("a", "b", True)

    mock_file.assert_called_once_with("a", "b", True)
    mock_untar.assert_called_once_with("foo", cleanup=True)
    if os.path.exists('a'):
        shutil.rmtree('a')
