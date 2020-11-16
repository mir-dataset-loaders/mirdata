# -*- coding: utf-8 -*-
import pytest
import smtplib

from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate

def send_mail(send_from, send_to, subject, text, files=None,
              server="127.0.0.1"):
    assert isinstance(send_to, list)

    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = COMMASPACE.join(send_to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(text))

    for f in files or []:
        with open(f, "rb") as fil:
            part = MIMEApplication(
                fil.read(),
                Name=basename(f)
            )
        # After the file is closed
        part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
        msg.attach(part)


    smtp = smtplib.SMTP(server)
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.close()


def pytest_addoption(parser):
    parser.addoption(
        "--local", action="store_true", default=False, help="run local tests"
    )
    parser.addoption("--dataset", type=str, default="", help="dataset to test locally")
    parser.addoption(
        "--skip-download", action="store_true", default=False, help="skip download step"
    )
    parser.addoption(
        "--send-email", action="store_true", default=False, help="send test results on email"
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

@pytest.fixture(scope='session')
def send_email(request):
    return request.config.getoption('--send-email')


def pytest_sessionstart(session):
    session.results = dict()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    result = outcome.get_result()

    if result.when == 'call':
        item.session.results[item] = result

def pytest_sessionfinish(session, exitstatus):
    print()
    print('run status code:', exitstatus)
    passed_amount = sum(1 for result in session.results.values() if result.passed)
    failed_amount = sum(1 for result in session.results.values() if result.failed)
    print(f'there are {passed_amount} passed and {failed_amount} failed tests')
    if session.config.option.send_email:
        send_mail('mirdatatests@gmail.com', ['miron.marius@upf.edu'], 'mirdata weekly reports', 'My Text', server="myserver")
