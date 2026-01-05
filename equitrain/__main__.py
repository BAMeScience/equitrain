"""Module entrypoint for Equitrain CLI helpers."""

from __future__ import annotations

import importlib
import sys

_COMMANDS = {
    'train': 'equitrain.scripts.equitrain',
    'preprocess': 'equitrain.scripts.equitrain_preprocess',
    'predict': 'equitrain.scripts.equitrain_predict',
    'evaluate': 'equitrain.scripts.equitrain_evaluate',
    'export': 'equitrain.scripts.equitrain_export',
    'inspect': 'equitrain.scripts.equitrain_inspect',
    'hdf5-benchmark': 'equitrain.scripts.equitrain_hdf5_benchmark',
    'hdf5-info': 'equitrain.scripts.equitrain_hdf5_info',
}


def _run_command(module_path: str, argv: list[str]) -> None:
    module = importlib.import_module(module_path)
    main = getattr(module, 'main', None)
    if main is None:
        raise SystemExit(f'Module {module_path} does not define a main().')
    sys.argv = [sys.argv[0]] + argv
    main()


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] in _COMMANDS:
        command = argv.pop(0)
        _run_command(_COMMANDS[command], argv)
        return
    _run_command(_COMMANDS['train'], argv)


if __name__ == '__main__':
    main()
