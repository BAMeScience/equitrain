import matgl
import torch
from matgl.models import M3GNet

from equitrain.data import Statistics


def get_statistics(filename='../../data/alexandria+mptraj-statistics.json'):
    """
    Load statistics from a file.

    Parameters
    ----------
    filename : str, optional
        Path to the statistics file. Default is '../../data/alexandria+mptraj-statistics.json'.

    Returns
    -------
    Statistics
        Statistics object containing atomic numbers, mean, std, etc.
    """
    print(f'Reading statistics from `{filename}`')

    statistics = Statistics.load(filename)

    return statistics


def get_element_types(statistics):
    """
    Get element types from statistics.

    Parameters
    ----------
    statistics : Statistics
        Statistics object containing atomic numbers.

    Returns
    -------
    list
        List of element symbols.
    """
    from equitrain.data.atomic import AtomicNumberTable

    # Convert atomic numbers to element symbols
    element_types = [
        AtomicNumberTable.from_number(num) for num in statistics.atomic_numbers
    ]

    return element_types


def create_m3gnet_model(element_types, r_max=5.0):
    """
    Create a new M3GNet model.

    Parameters
    ----------
    element_types : list
        List of element symbols to include in the model.
    r_max : float, optional
        Cutoff radius for the model. Default is 5.0.

    Returns
    -------
    M3GNet
        Initialized M3GNet model.
    """
    model = M3GNet(
        element_types=tuple(element_types),
        cutoff=r_max,
        threebody_cutoff=min(4.0, r_max),
        nblocks=3,
        max_n=3,
        max_l=3,
        units=64,
        ntargets=1,
    )

    return model


def main():
    """
    Main function to create and save an initial M3GNet model.
    """
    # Get statistics
    statistics = get_statistics()

    # Get element types from statistics
    element_types = get_element_types(statistics)

    # Create a new M3GNet model
    model = create_m3gnet_model(element_types, r_max=statistics.r_max)

    # Save the model
    torch.save(model, 'm3gnet-initial-model.pt')
    print('Model saved to m3gnet-initial-model.pt')


if __name__ == '__main__':
    main()
