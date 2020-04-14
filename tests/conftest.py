# -*- coding: utf-8 -*-
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--local", action="store_true", default=False, help="run local tests"
    )
    parser.addoption("--dataset", type=str, default="", help="dataset to test locally")
    parser.addoption(
        "--skip-download", action="store_true", default=False, help="skip download step"
    )


# @pytest.fixture(scope='session')
# def local(request):
#     return request.config.getoption('--local')


@pytest.fixture(scope='session')
def skip_local(request):
    if request.config.getoption('--local'):
        pytest.skip()


@pytest.fixture(scope='session')
def skip_remote(request):
    if not request.config.getoption('--local'):
        pytest.skip()


@pytest.fixture(scope='session')
def test_dataset(request):
    return request.config.getoption('--dataset')


@pytest.fixture(scope='session')
def skip_download(request):
    return request.config.getoption('--skip-download')
