from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Iterator, Sequence

import torch
import torch_geometric
from accelerate import Accelerator
from accelerate.data_loader import prepare_data_loader as prepare_accelerate_data_loader
from torch.utils.data import Sampler


class ReactionMetadataDataset(torch.utils.data.Dataset):
    """Attach loader-local reaction group ids to graph objects."""

    def __init__(self, dataset, reaction_group_ids: Sequence[int]):
        self.dataset = dataset
        self.reaction_group_ids = [int(group_id) for group_id in reaction_group_ids]
        if len(self.reaction_group_ids) != len(dataset):
            raise ValueError('Reaction group id count must match dataset length.')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        index = int(index)
        data = self.dataset[index]
        data.idx = index
        data.reaction_group_id = torch.tensor(
            self.reaction_group_ids[index], dtype=torch.long
        )
        return data


class ReactionGroupBatchSampler(Sampler[list[int]]):
    """Yield index batches that never split a reaction group."""

    def __init__(
        self,
        reaction_group_ids: Sequence[int],
        *,
        batch_size: int,
        shuffle: bool = False,
        generator: torch.Generator | None = None,
        num_replicas: int = 1,
        rank: int = 0,
        drop_last: bool = False,
        seed: int = 0,
    ):
        if batch_size is None or int(batch_size) <= 0:
            raise ValueError('A positive batch size is required for reaction grouping.')
        if int(num_replicas) <= 0:
            raise ValueError('num_replicas must be positive.')
        if int(rank) < 0 or int(rank) >= int(num_replicas):
            raise ValueError('rank must satisfy 0 <= rank < num_replicas.')
        self.max_batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.generator = generator
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        self.units = _reaction_units(reaction_group_ids)
        self.batches = _pack_reaction_units(self.units, self.max_batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        batch_indices = list(range(len(self.batches)))
        if self.shuffle:
            generator = self.generator
            if generator is None:
                generator = torch.Generator()
                generator.manual_seed(self.seed + self.epoch)
            permutation = torch.randperm(
                len(batch_indices), generator=generator
            ).tolist()
            batch_indices = [batch_indices[index] for index in permutation]

        if self.num_replicas > 1:
            batch_indices = self._make_even_batch_indices(batch_indices)
            batch_indices = batch_indices[self.rank :: self.num_replicas]

        for batch_index in batch_indices:
            yield list(self.batches[batch_index])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        batch_count = len(self.batches)
        if self.num_replicas <= 1:
            return batch_count
        if self.drop_last:
            return batch_count // self.num_replicas
        return (batch_count + self.num_replicas - 1) // self.num_replicas

    def _make_even_batch_indices(self, batch_indices: list[int]) -> list[int]:
        if not batch_indices:
            return []
        remainder = len(batch_indices) % self.num_replicas
        if remainder == 0:
            return batch_indices
        if self.drop_last:
            return batch_indices[: len(batch_indices) - remainder]

        needed = self.num_replicas - remainder
        padding = [batch_indices[index % len(batch_indices)] for index in range(needed)]
        return [*batch_indices, *padding]


class ReactionGraphCollater:
    def __init__(self, collate_fn, max_nodes=None, max_edges=None, drop=False):
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.drop = drop
        self.collate_fn = collate_fn

    def __call__(self, batch):
        dynamic_batches = []
        current_batch = []
        current_node_sum = 0
        current_edge_sum = 0

        for unit in _atomic_units(batch):
            unit_node_sum = sum(item.num_nodes for item in unit)
            unit_edge_sum = sum(item.num_edges for item in unit)
            group_id = _reaction_group_id(unit[0])
            grouped_reaction = group_id is not None and group_id >= 0

            if grouped_reaction:
                _raise_if_group_exceeds_limits(
                    group_id,
                    unit_node_sum,
                    unit_edge_sum,
                    self.max_nodes,
                    self.max_edges,
                )
            elif self._drop_oversized(unit_node_sum, unit_edge_sum):
                continue

            if current_batch:
                if (
                    self.max_nodes is not None
                    and current_node_sum + unit_node_sum > self.max_nodes
                ):
                    dynamic_batches.append(self.collate_fn(current_batch))
                    current_batch = []
                    current_node_sum = 0
                    current_edge_sum = 0

                if (
                    self.max_edges is not None
                    and current_edge_sum + unit_edge_sum > self.max_edges
                ):
                    dynamic_batches.append(self.collate_fn(current_batch))
                    current_batch = []
                    current_node_sum = 0
                    current_edge_sum = 0

            current_batch.extend(unit)
            current_node_sum += unit_node_sum
            current_edge_sum += unit_edge_sum

        if current_batch:
            dynamic_batches.append(self.collate_fn(current_batch))

        return dynamic_batches

    def _drop_oversized(self, node_count: int, edge_count: int) -> bool:
        if self.max_edges is not None and self.drop and edge_count > self.max_edges:
            return True
        if self.max_nodes is not None and self.drop and node_count > self.max_nodes:
            return True
        return False


class ReactionGraphLoader(torch_geometric.loader.DataLoader):
    def __init__(
        self,
        *args,
        max_nodes=None,
        max_edges=None,
        drop=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.collate_fn = ReactionGraphCollater(
            self.collate_fn, max_nodes=max_nodes, max_edges=max_edges, drop=drop
        )


def relative_reaction_losses_enabled(args) -> bool:
    return (
        float(getattr(args, 'barrier_weight', 0.0) or 0.0) != 0.0
        or float(getattr(args, 'reaction_energy_weight', 0.0) or 0.0) != 0.0
    )


def prepare_reaction_dataset(args, dataset, *, label: str):
    metadata = _reaction_metadata(dataset)
    _validate_relative_reaction_metadata(args, metadata, label=label)
    reaction_group_ids = _reaction_group_ids_from_metadata(metadata)
    return ReactionMetadataDataset(dataset, reaction_group_ids), reaction_group_ids


def get_reaction_loader(
    args,
    dataset,
    reaction_group_ids: Sequence[int],
    *,
    pin_memory: bool,
    num_workers: int,
    accelerator: Accelerator | None,
):
    data_loader = ReactionGraphLoader(
        dataset=dataset,
        batch_sampler=ReactionGroupBatchSampler(
            reaction_group_ids,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_replicas=int(getattr(accelerator, 'num_processes', 1) or 1),
            rank=int(getattr(accelerator, 'process_index', 0) or 0),
            seed=int(getattr(args, 'seed', 0) or 0),
        ),
        pin_memory=pin_memory,
        num_workers=num_workers,
        max_nodes=args.batch_max_nodes,
        max_edges=args.batch_max_edges,
        drop=args.batch_drop,
    )
    if accelerator is None:
        return data_loader
    return _prepare_rank_sharded_reaction_loader(data_loader, accelerator)


def _reaction_metadata(dataset) -> list[tuple[int, int, int]]:
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        result: list[tuple[int, int, int]] = []
        for child in dataset.datasets:
            result.extend(_reaction_metadata(child))
        return result

    reaction_metadata = getattr(dataset, 'reaction_metadata', None)
    if not callable(reaction_metadata):
        return [(0, -1, -1) for _ in range(len(dataset))]

    return [
        (int(source_id), int(reaction_id), int(state_id))
        for source_id, reaction_id, state_id in reaction_metadata()
    ]


def _reaction_group_ids(dataset) -> list[int]:
    return _reaction_group_ids_from_metadata(_reaction_metadata(dataset))


def _reaction_group_ids_from_metadata(
    reaction_metadata: list[tuple[int, int, int]],
) -> list[int]:
    key_to_group_id: dict[tuple[int, int], int] = {}
    group_ids = []
    next_group_id = 0
    for source_id, reaction_id, _state_id in reaction_metadata:
        reaction_id = int(reaction_id)
        if reaction_id < 0:
            group_ids.append(-1)
            continue
        key = (int(source_id), reaction_id)
        if key not in key_to_group_id:
            key_to_group_id[key] = next_group_id
            next_group_id += 1
        group_ids.append(key_to_group_id[key])
    return group_ids


def _validate_relative_reaction_metadata(
    args,
    reaction_metadata: list[tuple[int, int, int]],
    *,
    label: str,
) -> None:
    roles_by_reaction: dict[tuple[int, int], set[int]] = {}
    for source_id, reaction_id, state_id in reaction_metadata:
        if reaction_id < 0:
            continue
        roles_by_reaction.setdefault((source_id, reaction_id), set()).add(state_id)

    if getattr(args, 'barrier_weight', 0.0) > 0.0:
        _validate_relative_role_coverage(
            roles_by_reaction,
            required_roles={0, 1},
            option='--barrier-weight',
            label=label,
        )
    if getattr(args, 'reaction_energy_weight', 0.0) > 0.0:
        _validate_relative_role_coverage(
            roles_by_reaction,
            required_roles={0, 2},
            option='--reaction-energy-weight',
            label=label,
        )


def _validate_relative_role_coverage(
    roles_by_reaction: dict[tuple[int, int], set[int]],
    *,
    required_roles: set[int],
    option: str,
    label: str,
) -> None:
    complete = sum(
        1 for roles in roles_by_reaction.values() if required_roles.issubset(roles)
    )
    if complete == 0:
        roles = ', '.join(str(role) for role in sorted(required_roles))
        raise ValueError(
            f'No complete reaction groups found for {option} in {label}; '
            f'expected reaction_id >= 0 frames with state_id roles {roles}.'
        )

    incomplete = len(roles_by_reaction) - complete
    if incomplete > 0:
        warnings.warn(
            f'{incomplete} reaction groups in {label} are missing roles required by '
            f'{option} and will be skipped for that relative loss.',
            RuntimeWarning,
            stacklevel=3,
        )


def _prepare_rank_sharded_reaction_loader(data_loader, accelerator: Accelerator):
    rng_types = getattr(accelerator, 'rng_types', None)
    if rng_types is not None:
        rng_types = list(rng_types)
    return prepare_accelerate_data_loader(
        data_loader,
        device=accelerator.device,
        num_processes=1,
        process_index=0,
        split_batches=False,
        put_on_device=True,
        rng_types=rng_types,
        dispatch_batches=False,
        even_batches=False,
        non_blocking=getattr(accelerator, 'non_blocking', False),
        use_stateful_dataloader=getattr(accelerator, 'use_stateful_dataloader', False),
    )


def _pack_reaction_units(
    units: Sequence[Sequence[int]], max_batch_size: int
) -> list[list[int]]:
    batches: list[list[int]] = []
    batch: list[int] = []
    batch_count = 0
    for unit in units:
        unit = list(unit)
        unit_count = len(unit)
        if batch and batch_count + unit_count > max_batch_size:
            batches.append(batch)
            batch = []
            batch_count = 0
        batch.extend(unit)
        batch_count += unit_count
    if batch:
        batches.append(batch)
    return batches


def _reaction_units(reaction_group_ids: Sequence[int]) -> list[list[int]]:
    grouped: OrderedDict[int, list[int]] = OrderedDict()
    units: list[list[int]] = []
    for index, group_id in enumerate(reaction_group_ids):
        group_id = int(group_id)
        if group_id >= 0:
            if group_id not in grouped:
                grouped[group_id] = []
                units.append(grouped[group_id])
            grouped[group_id].append(index)
        else:
            units.append([index])
    return units


def _atomic_units(batch) -> list[list]:
    units_by_key: OrderedDict[tuple[str, int], list] = OrderedDict()
    ordinary_count = 0
    for item in batch:
        group_id = _reaction_group_id(item)
        if group_id is not None and group_id >= 0:
            key = ('reaction', group_id)
        else:
            key = ('ordinary', ordinary_count)
            ordinary_count += 1
        units_by_key.setdefault(key, []).append(item)
    return list(units_by_key.values())


def _reaction_group_id(item) -> int | None:
    value = getattr(item, 'reaction_group_id', None)
    if value is None:
        value = getattr(item, 'reaction_id', None)
    if value is None:
        return None
    if hasattr(value, 'detach'):
        value = value.detach().reshape(-1)[0].item()
    return int(value)


def _raise_if_group_exceeds_limits(
    group_id: int,
    node_count: int,
    edge_count: int,
    max_nodes: int | None,
    max_edges: int | None,
) -> None:
    if max_nodes is not None and node_count > max_nodes:
        raise ValueError(
            f'Reaction group {group_id} has {node_count} nodes, exceeding '
            f'--batch-max-nodes={max_nodes}; reaction groups cannot be split.'
        )
    if max_edges is not None and edge_count > max_edges:
        raise ValueError(
            f'Reaction group {group_id} has {edge_count} edges, exceeding '
            f'--batch-max-edges={max_edges}; reaction groups cannot be split.'
        )


__all__ = [
    'ReactionGraphCollater',
    'ReactionGraphLoader',
    'ReactionGroupBatchSampler',
    'ReactionMetadataDataset',
    'get_reaction_loader',
    'prepare_reaction_dataset',
    'relative_reaction_losses_enabled',
]
