import os
from pathlib import Path

import pytest


@pytest.fixture(scope='session')
def mace_model_path():
    path = Path(__file__).resolve().parents[2] / 'tests' / 'mace.model'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

if os.getenv('_PYTEST_RAISE', '0') != '0':

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value
