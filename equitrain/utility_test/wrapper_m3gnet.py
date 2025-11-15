from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from shutil import copy2

import torch
from ase.data import chemical_symbols

from equitrain.backends.torch_wrappers.m3gnet import M3GNetWrapper as TorchM3GNetWrapper

try:  # pragma: no cover - optional dependency guard
    from matgl.config import DEFAULT_ELEMENTS
    from matgl.models import M3GNet as MatGLM3GNet
except Exception as exc:  # pragma: no cover
    raise ImportError(
        'The test M3GNet wrapper requires `matgl` (and `dgl`) to be available.'
    ) from exc


class M3GNetWrapper(TorchM3GNetWrapper):
    """
    Test-friendly MatGL M3GNet wrapper with sensible defaults for the bundled fixtures.
    """

    def __init__(
        self,
        args,
        statistics_file: str | Path | None = None,
        element_types: Iterable[str | int] | None = None,
    ):
        statistics_path = self._resolve_statistics_path(statistics_file)
        stats_numbers, stats_energies = self._load_statistics(statistics_path)

        resolved_symbols = self._resolve_statistics_elements(
            element_types, stats_numbers
        )
        model = MatGLM3GNet(element_types=tuple(resolved_symbols))

        super().__init__(args, model=model, element_types=resolved_symbols)

        if stats_energies is not None:
            energies = torch.tensor(
                [float(stats_energies.get(str(z), 0.0)) for z in self.atomic_numbers],
                dtype=self.atomic_energies.dtype,
                device=self.atomic_energies.device,
            )
            self.atomic_energies.copy_(energies)

        self._ensure_test_resources()

    @staticmethod
    def _resolve_statistics_path(statistics_file: str | Path | None) -> Path | None:
        if statistics_file is not None:
            candidate = Path(statistics_file).expanduser().resolve()
            return candidate if candidate.is_file() else None

        repo_root = Path(__file__).resolve().parents[2]
        candidate = repo_root / 'tests' / 'data' / 'statistics.json'
        return candidate if candidate.is_file() else None

    @staticmethod
    def _load_statistics(
        path: Path | None,
    ) -> tuple[list[int] | None, dict[str, float] | None]:
        if path is None or not path.is_file():
            return None, None

        with path.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)

        numbers = payload.get('atomic_numbers')
        energies = payload.get('atomic_energies')

        if numbers is not None:
            numbers = [int(z) for z in numbers]

        return numbers, energies

    @staticmethod
    def _resolve_statistics_elements(
        element_types: Iterable[str | int] | None,
        statistics_numbers: list[int] | None,
    ) -> list[str]:
        if element_types is not None:
            return [str(symbol).strip() for symbol in element_types]

        if statistics_numbers is not None:
            symbols: list[str] = []
            for number in statistics_numbers:
                if number < len(chemical_symbols):
                    symbols.append(chemical_symbols[number])
            return symbols

        return list(DEFAULT_ELEMENTS)

    @staticmethod
    def _ensure_test_resources() -> None:
        repo_root = Path(__file__).resolve().parents[2]
        source_data_dir = repo_root / 'tests' / 'data'
        target_data_dir = repo_root / 'data'
        target_data_dir.mkdir(parents=True, exist_ok=True)

        for filename in ('train.h5', 'valid.h5'):
            source = source_data_dir / filename
            target = target_data_dir / filename
            if source.is_file() and not target.exists():
                copy2(source, target)

        xyz_source = repo_root / 'tests' / 'data.xyz'
        xyz_target = repo_root / 'data.xyz'
        if xyz_source.is_file() and not xyz_target.exists():
            copy2(xyz_source, xyz_target)
