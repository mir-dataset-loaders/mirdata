from __future__ import absolute_import

import os
import sys

from mirdata import web_downloader

import pytest

if sys.version_info.major == 3:
    builtin_module_name = 'builtins'
else:
    builtin_module_name = '__builtin__'


@pytest.fixture
def mock_download(mocker):
    return mocker.patch.object(web_downloader, 'download_from_remote')


@pytest.fixture
def mock_untar(mocker):
    return mocker.patch.object(web_downloader, 'untar')


@pytest.fixture
def mock_unzip(mocker):
    return mocker.patch.object(web_downloader, 'unzip')


@pytest.fixture
def mock_force_delete_all(mocker):
    return mocker.patch.object(web_downloader, 'force_delete_all')


def test_download_from_remote(httpserver, tmpdir):
    httpserver.serve_content(open('tests/resources/remote.wav').read())

    TEST_REMOTE = web_downloader.RemoteFileMetadata(
        filename='remote.wav',
        url=httpserver.url,
        checksum=('3f77d0d69dc41b3696f074ad6bf2852f')
    )

    download_path = web_downloader.download_from_remote(TEST_REMOTE, str(tmpdir))
    expected_download_path = os.path.join(str(tmpdir), 'remote.wav')
    assert expected_download_path == download_path


def test_download_from_remote_raises_IOError(httpserver, tmpdir):
    httpserver.serve_content('File not found!', 404)

    TEST_REMOTE = web_downloader.RemoteFileMetadata(
        filename='remote.wav',
        url=httpserver.url,
        checksum=('1234')
    )

    with pytest.raises(IOError):
        web_downloader.download_from_remote(TEST_REMOTE, str(tmpdir))


def test_unzip(tmpdir):
    web_downloader.unzip('tests/resources/remote.zip', str(tmpdir))

    expected_file_location = os.path.join(str(tmpdir), 'remote.wav')
    assert os.path.exists(expected_file_location)


def test_untar(tmpdir):
    web_downloader.untar('tests/resources/remote.tar.gz', str(tmpdir))

    expected_file_location = os.path.join(str(tmpdir), 'remote.wav')
    assert os.path.exists(expected_file_location)


def test_force_delete_all_nonempty_data_home(httpserver, tmpdir):
    tmpdir_str = str(tmpdir)
    remote_filename = 'remote.wav'
    TEST_REMOTE = web_downloader.RemoteFileMetadata(
        filename=remote_filename,
        url=httpserver.url,
        checksum=('1234')
    )

    with pytest.raises(IOError):
        web_downloader.download_from_remote(TEST_REMOTE, tmpdir_str)

    web_downloader.untar('tests/resources/remote.tar.gz', tmpdir_str)
    assert os.path.exists(os.path.join(tmpdir_str, remote_filename))
    assert os.path.exists(tmpdir_str)
    web_downloader.force_delete_all(TEST_REMOTE, tmpdir_str)
    assert not os.path.exists(os.path.join(tmpdir_str, remote_filename))
    assert not os.path.exists(tmpdir_str)
