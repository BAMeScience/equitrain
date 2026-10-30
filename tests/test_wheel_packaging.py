from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def test_wheel_contains_subpackages_and_cli_entrypoint_modules(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    source = tmp_path / 'source'
    source.mkdir()
    for name in ('pyproject.toml', 'README.md', 'LICENSE'):
        shutil.copy(repo / name, source / name)
    shutil.copytree(
        repo / 'equitrain',
        source / 'equitrain',
        ignore=shutil.ignore_patterns('__pycache__'),
    )
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'from setuptools.build_meta import build_wheel; build_wheel("dist")',
        ],
        cwd=source,
        env={**os.environ, 'SETUPTOOLS_SCM_PRETEND_VERSION': '0.0.0'},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheel = next((source / 'dist').glob('*.whl'))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        expected = {
            f'equitrain/{path.relative_to(repo / "equitrain").as_posix()}'
            for path in (repo / 'equitrain').rglob('*.py')
        }
        assert expected <= names
        installed = tmp_path / 'installed'
        archive.extractall(installed)
    result = subprocess.run(
        [
            sys.executable,
            '-S',
            '-c',
            'import importlib.util; '
            'assert importlib.util.find_spec("equitrain.scripts.equitrain"); '
            'assert importlib.util.find_spec("equitrain.backends.torch_backend")',
        ],
        cwd=tmp_path,
        env={**os.environ, 'PYTHONPATH': str(installed)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
