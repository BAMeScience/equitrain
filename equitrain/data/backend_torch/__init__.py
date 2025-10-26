from importlib import import_module

_EXPORT_MAP = {
    'dataloader_update_errors': '.loaders',
    'get_dataloader': '.loaders',
    'get_dataloaders': '.loaders',
    'AtomsToGraphs': '.atoms_to_graphs',
    'scatter_sum': '.scatter',
    'scatter_mean': '.scatter',
    'scatter_std': '.scatter',
    'to_numpy': '.utility',
    'atomic_numbers_to_indices': '.utility',
    'to_one_hot': '.utility',
    'compute_one_hot': '.utility',
    'AtomicEnergiesBlock': '.statistics',
    'AtomicNumberTable': '.statistics',
    'compute_statistics': '.statistics',
    'compute_atomic_numbers': '.statistics',
    'compute_average_atomic_energies': '.statistics',
}

__all__ = list(_EXPORT_MAP.keys())


def __getattr__(name):
    try:
        module = import_module(f'{__name__}{_EXPORT_MAP[name]}')
    except KeyError as exc:
        raise AttributeError(name) from exc
    return getattr(module, name)
