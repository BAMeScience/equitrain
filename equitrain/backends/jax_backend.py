from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
from flax import serialization
from mace_jax.cli import mace_torch2jax
from mace_jax.data.utils import (
    AtomicNumberTable as JaxAtomicNumberTable,
)
from mace_jax.data.utils import (
    Configuration as JaxConfiguration,
)
from mace_jax.data.utils import (
    GraphDataLoader,
    graph_from_configuration,
)

from equitrain.argparser import ArgsFormatter, ArgumentError
from equitrain.backends.common import (
    ensure_output_dir,
    init_logger,
    validate_evaluate_args,
    validate_training_args,
)
from equitrain.data.configuration import Configuration as EqConfiguration
from equitrain.data.format_hdf5.dataset import HDF5Dataset

_DEFAULT_CONFIG_NAME = 'config.json'
_DEFAULT_PARAMS_NAME = 'params.msgpack'


@dataclass(frozen=True)
class ModelBundle:
    config: dict
    params: dict
    module: object


def _set_jax_dtype(dtype: str) -> None:
    dtype = (dtype or 'float32').lower()
    if dtype == 'float64':
        jax.config.update('jax_enable_x64', True)
    elif dtype in {'float32', 'float16'}:
        jax.config.update('jax_enable_x64', False)
    else:
        raise ArgumentError(f'Unsupported dtype for JAX backend: {dtype}')


def _resolve_model_paths(model_arg: str) -> tuple[Path, Path]:
    path = Path(model_arg).expanduser().resolve()

    if path.is_dir():
        config_path = path / _DEFAULT_CONFIG_NAME
        params_path = path / _DEFAULT_PARAMS_NAME
    elif path.suffix == '.json':
        config_path = path
        params_path = path.with_suffix('.msgpack')
    else:
        params_path = path
        config_path = path.with_suffix('.json')

    if not config_path.exists():
        raise FileNotFoundError(
            f'Unable to locate JAX model configuration at {config_path}'
        )
    if not params_path.exists():
        raise FileNotFoundError(
            f'Unable to locate serialized JAX parameters at {params_path}'
        )

    return config_path, params_path


def _load_model_bundle(model_arg: str, dtype: str) -> ModelBundle:
    config_path, params_path = _resolve_model_paths(model_arg)
    config = json.loads(config_path.read_text())

    _set_jax_dtype(dtype)

    jax_module = mace_torch2jax._build_jax_model(config)
    template = mace_torch2jax._prepare_template_data(config)
    variables = jax_module.init(jax.random.PRNGKey(0), template)
    variables = serialization.from_bytes(variables, params_path.read_bytes())

    return ModelBundle(config=config, params=variables, module=jax_module)


def _voigt_to_full(stress: np.ndarray | None) -> np.ndarray | None:
    if stress is None:
        return None
    stress = np.asarray(stress)
    if stress.shape == (3, 3):
        return stress
    if stress.shape != (6,):
        return None

    sxx, syy, szz, syz, sxz, sxy = stress
    return np.array(
        [
            [sxx, sxy, sxz],
            [sxy, syy, syz],
            [sxz, syz, szz],
        ],
        dtype=stress.dtype,
    )


def _atoms_to_graphs(
    data_path: Path | str,
    r_max: float,
    z_table: JaxAtomicNumberTable,
) -> list:
    if data_path is None:
        return []

    dataset = HDF5Dataset(data_path, mode='r')
    graphs: list = []
    try:
        for idx in range(len(dataset)):
            atoms = dataset[idx]
            eq_conf = EqConfiguration.from_atoms(atoms)
            jax_conf = JaxConfiguration(
                atomic_numbers=eq_conf.atomic_numbers,
                positions=np.asarray(eq_conf.positions, dtype=np.float32),
                energy=np.array(eq_conf.energy, dtype=np.float32),
                forces=np.asarray(eq_conf.forces, dtype=np.float32),
                stress=_voigt_to_full(eq_conf.stress),
                cell=np.asarray(eq_conf.cell, dtype=np.float32),
                pbc=tuple(bool(x) for x in eq_conf.pbc),
                weight=np.array(eq_conf.energy_weight, dtype=np.float32),
            )
            graph = graph_from_configuration(
                jax_conf,
                cutoff=float(r_max),
                z_table=z_table,
            )
            graphs.append(graph)
    finally:
        dataset.close()
    return graphs


def _compute_padding_limits(
    graphs: Iterable,
    max_nodes_override: int | None,
    max_edges_override: int | None,
) -> tuple[int, int]:
    max_nodes = 0
    max_edges = 0
    for graph in graphs:
        max_nodes = max(max_nodes, int(graph.n_node.sum()))
        max_edges = max(max_edges, int(graph.n_edge.sum()))

    if max_nodes_override is not None:
        max_nodes = min(max_nodes, max_nodes_override)
    if max_edges_override is not None:
        max_edges = min(max_edges, max_edges_override)

    # GraphDataLoader expects strict inequality, hence +1
    return max_nodes + 1, max_edges + 1


def _graph_to_data_jax(
    graph: jraph.GraphsTuple, num_species: int
) -> dict[str, jnp.ndarray]:
    positions = jnp.asarray(graph.nodes.positions, dtype=jnp.float32)
    shifts = jnp.asarray(graph.edges.shifts, dtype=positions.dtype)
    cell = jnp.asarray(graph.globals.cell, dtype=positions.dtype)

    species = jnp.asarray(graph.nodes.species, dtype=jnp.int32)
    senders = jnp.asarray(graph.senders, dtype=jnp.int32)
    receivers = jnp.asarray(graph.receivers, dtype=jnp.int32)

    n_node = jnp.asarray(graph.n_node, dtype=jnp.int32)
    all_positive = jnp.all(n_node > 0)
    graph_mask = jnp.where(
        all_positive,
        jnp.ones_like(n_node, dtype=bool),
        jraph.get_graph_padding_mask(graph),
    )
    node_mask = jnp.repeat(
        graph_mask, graph.n_node, total_repeat_length=positions.shape[0]
    ).astype(positions.dtype)

    node_attrs = jax.nn.one_hot(
        species,
        num_classes=num_species,
        dtype=positions.dtype,
    )
    node_attrs = node_attrs * node_mask[:, None]

    graph_indices = jnp.arange(graph.n_node.shape[0], dtype=jnp.int32)
    batch = jnp.repeat(
        graph_indices, graph.n_node, total_repeat_length=positions.shape[0]
    )
    ptr = jnp.concatenate([
        jnp.array([0], dtype=jnp.int32),
        jnp.cumsum(graph.n_node.astype(jnp.int32)),
    ])

    data_dict: dict[str, jnp.ndarray] = {
        'positions': positions,
        'node_attrs': node_attrs,
        'edge_index': jnp.stack([senders, receivers], axis=0),
        'shifts': shifts,
        'batch': batch,
        'ptr': ptr,
        'cell': cell,
    }

    unit_shifts = getattr(graph.edges, 'unit_shifts', None)
    if unit_shifts is None:
        unit_shifts = jnp.zeros(shifts.shape, dtype=positions.dtype)
    else:
        unit_shifts = jnp.asarray(unit_shifts, dtype=positions.dtype)
    data_dict['unit_shifts'] = unit_shifts

    if hasattr(graph.nodes, 'head'):
        data_dict['head'] = graph.nodes.head

    return data_dict


def _build_loader(
    graphs: list,
    *,
    batch_size: int,
    shuffle: bool,
    max_nodes: int | None,
    max_edges: int | None,
) -> GraphDataLoader | None:
    if not graphs:
        return None

    pad_nodes, pad_edges = _compute_padding_limits(graphs, max_nodes, max_edges)

    return GraphDataLoader(
        graphs=graphs,
        n_node=pad_nodes,
        n_edge=pad_edges,
        n_graph=max(batch_size, 2),
        shuffle=shuffle,
    )


def _build_loss_fn(jax_module, z_table: JaxAtomicNumberTable, energy_weight: float):
    if energy_weight <= 0.0:
        raise ArgumentError(
            'The JAX backend currently requires a positive --energy-weight value.'
        )

    num_species = len(z_table)

    def loss_fn(variables, graph):
        mask = jraph.get_graph_padding_mask(graph).astype(jnp.float32)

        data_dict = _graph_to_data_jax(graph, num_species=num_species)
        outputs = jax_module.apply(
            variables,
            data_dict,
            compute_force=False,
            compute_stress=False,
        )

        pred_energy = jnp.reshape(outputs['energy'], mask.shape)
        target_energy = jnp.reshape(jnp.asarray(graph.globals.energy), mask.shape)
        weights = jnp.reshape(jnp.asarray(graph.globals.weight), mask.shape)

        diff = pred_energy - target_energy
        sq_error = diff * diff
        weighted = sq_error * weights * mask

        denom = jnp.maximum(jnp.sum(weights * mask), 1.0)
        return energy_weight * jnp.sum(weighted) / denom

    return loss_fn


def _train_loop(
    variables,
    optimizer,
    opt_state,
    train_loader: GraphDataLoader,
    loss_fn,
):
    grad_fn = jax.value_and_grad(loss_fn)

    @jax.jit
    def train_step(current_vars, current_opt_state, graph):
        loss, grads = grad_fn(current_vars, graph)
        updates, new_opt_state = optimizer.update(
            grads, current_opt_state, current_vars
        )
        new_vars = optax.apply_updates(current_vars, updates)
        return new_vars, new_opt_state, loss

    losses = []
    for graph in train_loader:
        variables, opt_state, loss = train_step(variables, opt_state, graph)
        losses.append(float(jax.device_get(loss)))

    return variables, opt_state, float(np.mean(losses)) if losses else 0.0


def _evaluate_loop(variables, loss_fn, loader: GraphDataLoader | None):
    if loader is None:
        return None

    eval_step = jax.jit(loss_fn)
    losses = []
    for graph in loader:
        loss = eval_step(variables, graph)
        losses.append(float(jax.device_get(loss)))

    return float(np.mean(losses)) if losses else None


def _save_parameters(output_dir: Path, variables) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    params_path = output_dir / 'jax_params.msgpack'
    params_path.write_bytes(serialization.to_bytes(variables))


def _ensure_forces_not_requested(args):
    if getattr(args, 'forces_weight', 0.0) not in (0.0, None):
        raise NotImplementedError(
            'The current JAX backend only supports energy training.'
        )
    if getattr(args, 'stress_weight', 0.0) not in (0.0, None):
        raise NotImplementedError(
            'The current JAX backend only supports energy training.'
        )


def train(args):
    validate_training_args(args, 'jax')

    _ensure_forces_not_requested(args)

    ensure_output_dir(getattr(args, 'output_dir', None))

    logger = init_logger(
        args,
        backend_name='jax',
        enable_logging=True,
        log_to_file=True,
        output_dir=args.output_dir,
    )
    logger.log(1, ArgsFormatter(args))

    bundle = _load_model_bundle(args.model, dtype=args.dtype)

    atomic_numbers = bundle.config.get('atomic_numbers')
    if not atomic_numbers:
        raise RuntimeError('Model configuration is missing `atomic_numbers`.')
    z_table = JaxAtomicNumberTable(atomic_numbers)

    r_max = float(bundle.config.get('r_max', 0.0))
    if r_max <= 0.0:
        raise RuntimeError('Model configuration must define a positive `r_max`.')

    train_graphs = _atoms_to_graphs(args.train_file, r_max, z_table)
    valid_graphs = _atoms_to_graphs(args.valid_file, r_max, z_table)

    if not train_graphs:
        raise RuntimeError('Training dataset is empty.')

    train_loader = _build_loader(
        train_graphs,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        max_nodes=args.batch_max_nodes,
        max_edges=args.batch_max_edges,
    )
    valid_loader = _build_loader(
        valid_graphs,
        batch_size=args.batch_size,
        shuffle=False,
        max_nodes=args.batch_max_nodes,
        max_edges=args.batch_max_edges,
    )

    loss_fn = _build_loss_fn(bundle.module, z_table, args.energy_weight)
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(bundle.params)

    num_epochs = args.epochs
    start_epoch = args.epochs_start

    best_val = None
    best_params = bundle.params
    train_loss = 0.0

    for epoch_offset in range(num_epochs):
        epoch = start_epoch + epoch_offset

        updated_params, opt_state, train_loss = _train_loop(
            bundle.params,
            optimizer,
            opt_state,
            train_loader,
            loss_fn,
        )
        bundle = ModelBundle(
            config=bundle.config, params=updated_params, module=bundle.module
        )

        val_loss = _evaluate_loop(bundle.params, loss_fn, valid_loader)

        logger.log(
            1,
            f'Epoch {epoch}: train_loss={train_loss:.6f}'
            + (f', val_loss={val_loss:.6f}' if val_loss is not None else ''),
        )

        if val_loss is None:
            best_params = bundle.params
        elif best_val is None or val_loss < best_val:
            best_val = val_loss
            best_params = bundle.params

    _save_parameters(Path(args.output_dir), best_params)

    return {'train_loss': train_loss, 'val_loss': best_val}


def evaluate(args):
    validate_evaluate_args(args, 'jax')

    _ensure_forces_not_requested(args)

    logger = init_logger(
        args,
        backend_name='jax',
        enable_logging=True,
        log_to_file=False,
        output_dir=None,
    )
    logger.log(1, ArgsFormatter(args))

    bundle = _load_model_bundle(args.model, dtype=args.dtype)

    atomic_numbers = bundle.config.get('atomic_numbers')
    if not atomic_numbers:
        raise RuntimeError('Model configuration is missing `atomic_numbers`.')
    z_table = JaxAtomicNumberTable(atomic_numbers)

    r_max = float(bundle.config.get('r_max', 0.0))
    if r_max <= 0.0:
        raise RuntimeError('Model configuration must define a positive `r_max`.')

    test_graphs = _atoms_to_graphs(args.test_file, r_max, z_table)
    if not test_graphs:
        raise RuntimeError('Test dataset is empty.')

    test_loader = _build_loader(
        test_graphs,
        batch_size=args.batch_size,
        shuffle=False,
        max_nodes=args.batch_max_nodes,
        max_edges=args.batch_max_edges,
    )

    loss_fn = _build_loss_fn(bundle.module, z_table, args.energy_weight)
    test_loss = _evaluate_loop(bundle.params, loss_fn, test_loader)

    logger.log(
        1,
        f'Test loss: {test_loss:.6f}'
        if test_loss is not None
        else 'No test loss computed',
    )
    return test_loss
