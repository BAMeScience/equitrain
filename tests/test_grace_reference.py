"""Optional TensorFlow parity using fresh local models, never pretrained weights.

Run with EQUITRAIN_GRACE_REFERENCE=/path/to/grace-tensorpotential.
"""

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from equitrain.models.grace import Grace1L, GraceConfig

REFERENCE = os.environ.get('EQUITRAIN_GRACE_REFERENCE')
pytestmark = pytest.mark.skipif(
    not REFERENCE, reason='Set EQUITRAIN_GRACE_REFERENCE to a local checkout.'
)


@pytest.fixture(scope='module')
def reference():
    import sys

    sys.path.insert(0, str(Path(REFERENCE).resolve()))
    # Set before importing TF, as required by the upstream package.
    os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    tf = pytest.importorskip('tensorflow')
    from tensorpotential import constants
    from tensorpotential.potentials.presets import GRACE_1LAYER_v2_25

    return tf, constants, GRACE_1LAYER_v2_25


def build_pair(reference, lmax, max_order):
    tf, _, preset = reference
    c = GraceConfig(
        (8, 1),
        r_max=3.0,
        lmax=lmax,
        max_order=max_order,
        n_rad_base=3,
        n_rad_max=3,
        prod_func_n_max=4,
        embedding_size=4,
        n_mlp_dens=3,
        avg_n_neigh=2.5,
        energy_scale=0.7,
        atomic_energies=(-1.0, -0.2),
    )
    torch.manual_seed(41)
    native = Grace1L(c).double()
    instructions = preset(
        element_map={'O': 0, 'H': 1},
        rcut=c.r_max,
        lmax=lmax,
        max_order=max_order,
        n_rad_base=c.n_rad_base,
        n_rad_max=c.n_rad_max,
        prod_func_n_max=c.prod_func_n_max,
        embedding_size=c.embedding_size,
        n_mlp_dens=c.n_mlp_dens,
        avg_n_neigh=c.avg_n_neigh,
        constant_out_scale=c.energy_scale,
        atomic_shift_map=dict(enumerate(c.atomic_energies)),
    ).get_instructions()
    for instruction in instructions.values():
        instruction.build(tf.float64)
    pairs = []

    def assign(variable, parameter):
        variable.assign(parameter.detach().numpy())
        pairs.append((variable, parameter))

    assign(instructions['Z'].w, native.embedding)
    assign(instructions['A'].lin_transform.w, native.chemical_projection.weight)
    for layer, native_layer in zip(
        instructions['R'].layers, native.radial.layers, strict=True
    ):
        assign(layer.w, native_layer.weight)
    for name, mixer in native.mixers.items():
        assign(instructions[name].w_left, mixer.weight_left)
        assign(instructions[name].w_right, mixer.weight_right)
    for name, reducer in native.reducers.items():
        assign(getattr(instructions['rho'], f'reducing_{name}'), reducer.weight)
    readout = instructions['LinMLPOut2ScalarTarget'].mlp
    for i, layer in enumerate(native.readout.layers):
        assign(getattr(readout, f'layer{i}').w, layer.weight)
    assign(
        instructions['TrainableShiftTarget'].at_shifts, native.trainable_shift[:, None]
    )
    # Keep the actual parameter in the gradient comparison (rather than its view).
    pairs[-1] = (pairs[-1][0], native.trainable_shift)
    return native, instructions, pairs


@pytest.mark.parametrize(
    'lmax,max_order',
    [(degree, 4) for degree in range(5)] + [(2, k) for k in range(1, 4)],
)
def test_reference_features_energy_derivatives_and_parameter_gradients(
    reference, lmax, max_order
):
    tf, constants, _ = reference
    native, instructions, pairs = build_pair(reference, lmax, max_order)
    positions = np.array([[0.2, 0.3, 0.4], [1.1, 0.4, 0.6], [0.4, 1.2, 0.7]])
    species = np.array([0, 1, 1], dtype=np.int32)
    centers = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    neighbors = np.array([1, 2, 0, 2, 0, 1], dtype=np.int32)
    vectors = positions[neighbors] - positions[centers]
    b_torch = torch.tensor(vectors, dtype=torch.float64, requires_grad=True)
    strain_torch = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
    transformed = b_torch + b_torch @ ((strain_torch + strain_torch.T) / 2)
    values = native.features(
        transformed,
        torch.tensor(species).long(),
        torch.tensor(centers).long(),
        torch.tensor(neighbors).long(),
    )
    energy = values['atomic_energy'].sum()
    bond_grad, virial_grad = torch.autograd.grad(
        energy, (b_torch, strain_torch), create_graph=True
    )
    objective = energy.square() + bond_grad.square().sum() + virial_grad.square().sum()
    native_gradients = torch.autograd.grad(objective, [p for _, p in pairs])
    b_tf = tf.Variable(vectors, dtype=tf.float64)
    strain_tf = tf.Variable(np.zeros((3, 3)), dtype=tf.float64)
    with tf.GradientTape() as outer:
        with tf.GradientTape(persistent=True) as inner:
            transformed_tf = b_tf + tf.matmul(
                b_tf, (strain_tf + tf.transpose(strain_tf)) / 2
            )
            data = {
                constants.BOND_VECTOR: transformed_tf,
                constants.ATOMIC_MU_I: tf.constant(species),
                constants.BOND_MU_I: tf.constant(species[centers]),
                constants.BOND_MU_J: tf.constant(species[neighbors]),
                constants.BOND_IND_I: tf.constant(centers),
                constants.BOND_IND_J: tf.constant(neighbors),
                constants.N_ATOMS_BATCH_REAL: tf.constant(3),
                constants.N_ATOMS_BATCH_TOTAL: tf.constant(3),
            }
            for instruction in instructions.values():
                instruction(data)
            energy_tf = tf.reduce_sum(data['atomic_energy'])
        bond_grad_tf = inner.gradient(energy_tf, b_tf)
        virial_grad_tf = inner.gradient(energy_tf, strain_tf)
        objective_tf = (
            energy_tf**2
            + tf.reduce_sum(bond_grad_tf**2)
            + tf.reduce_sum(virial_grad_tf**2)
        )
    reference_gradients = outer.gradient(objective_tf, [v for v, _ in pairs])
    for name, value in values.items():
        expected = data[name].numpy()
        if name == 'rho':
            expected = expected[..., 0]
        if name == 'atomic_energy':
            expected = expected[:, 0]
        np.testing.assert_allclose(
            value.detach().numpy(), expected, rtol=2e-10, atol=2e-11, err_msg=name
        )
    np.testing.assert_allclose(
        bond_grad.detach().numpy(), bond_grad_tf.numpy(), rtol=2e-9, atol=2e-10
    )
    np.testing.assert_allclose(
        virial_grad.detach().numpy(), virial_grad_tf.numpy(), rtol=2e-9, atol=2e-10
    )
    for actual, expected in zip(native_gradients, reference_gradients, strict=True):
        if isinstance(expected, tf.IndexedSlices):
            expected = tf.convert_to_tensor(expected)
        np.testing.assert_allclose(
            actual.detach().numpy().reshape(-1),
            expected.numpy().reshape(-1),
            rtol=2e-8,
            atol=2e-9,
        )
