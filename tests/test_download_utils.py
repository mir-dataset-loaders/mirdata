from __future__ import absolute_import

import os
import sys

from mirdata import download_utils

import pytest

if sys.version_info.major == 3:
    builtin_module_name = 'builtins'
else:
    builtin_module_name = '__builtin__'


@pytest.fixture
def mock_download(mocker):
    return mocker.patch.object(download_utils, 'download_from_remote')


@pytest.fixture
def mock_untar(mocker):
    return mocker.patch.object(download_utils, 'untar')


@pytest.fixture
def mock_unzip(mocker):
    return mocker.patch.object(download_utils, 'unzip')


def test_download_from_remote(httpserver, tmpdir):
    httpserver.serve_content(open('tests/resources/remote.wav').read())

    TEST_REMOTE = download_utils.RemoteFileMetadata(
        filename='remote.wav',
        url=httpserver.url,
        checksum=('3f77d0d69dc41b3696f074ad6bf2852f')
    )

    download_path = download_utils.download_from_remote(TEST_REMOTE, str(tmpdir))
    expected_download_path = os.path.join(str(tmpdir), 'remote.wav')
    assert expected_download_path == download_path


def test_download_from_remote_raises_IOError(httpserver, tmpdir):
    httpserver.serve_content('File not found!', 404)

    TEST_REMOTE = download_utils.RemoteFileMetadata(
        filename='remote.wav',
        url=httpserver.url,
        checksum=('1234')
    )

    with pytest.raises(IOError):
        download_utils.download_from_remote(TEST_REMOTE, str(tmpdir))


def test_unzip(tmpdir):
    download_utils.unzip('tests/resources/remote.zip', str(tmpdir))

    expected_file_location = os.path.join(str(tmpdir), 'remote.wav')
    assert os.path.exists(expected_file_location)


def test_untar(tmpdir):
    download_utils.untar('tests/resources/remote.tar.gz', str(tmpdir))

    expected_file_location = os.path.join(str(tmpdir), 'remote.wav')
    assert os.path.exists(expected_file_location)
