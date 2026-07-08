import jax
import jax.numpy as jnp
import pytest
from flax.linen.initializers import orthogonal, constant

from gtd_algos.src.agents.value_networks import (
    DenseQNetworkContinuousAction,
    DenseDoubleQNetworkContinuousAction,
)

OBS_DIM = 8
ACTION_DIM = 2
HIDDENS = (256, 256)
BATCH_SIZE = 4


@pytest.fixture
def kernel_init():
    return orthogonal(jnp.sqrt(2))


@pytest.fixture
def single_q(kernel_init):
    return DenseQNetworkContinuousAction(
        hiddens=HIDDENS,
        layer_norm=False,
        activation="relu",
        kernel_init=kernel_init,
    )


@pytest.fixture
def double_q(kernel_init):
    return DenseDoubleQNetworkContinuousAction(
        hiddens=HIDDENS,
        layer_norm=False,
        activation="relu",
        kernel_init=kernel_init,
    )


@pytest.fixture
def batched_inputs():
    obs = jnp.ones((BATCH_SIZE, OBS_DIM))
    action = jnp.ones((BATCH_SIZE, ACTION_DIM))
    return obs, action


@pytest.fixture
def single_inputs():
    obs = jnp.ones(OBS_DIM)
    action = jnp.ones(ACTION_DIM)
    return obs, action


@pytest.fixture
def single_q_params(single_q, batched_inputs):
    obs, action = batched_inputs
    return single_q.init(jax.random.PRNGKey(0), obs, action)


@pytest.fixture
def double_q_params(double_q, batched_inputs):
    obs, action = batched_inputs
    return double_q.init(jax.random.PRNGKey(0), obs, action)


# ---------------------------------------------------------------------------
# DenseQNetworkContinuousAction
# ---------------------------------------------------------------------------

def test_single_q_batched_output_shape(single_q, single_q_params, batched_inputs):
    obs, action = batched_inputs
    q = single_q.apply(single_q_params, obs, action)
    assert q.shape == (BATCH_SIZE,), f"expected ({BATCH_SIZE},), got {q.shape}"


def test_single_q_unbatched_output_is_scalar(single_q, single_q_params, single_inputs):
    obs, action = single_inputs
    q = single_q.apply(single_q_params, obs, action)
    assert q.shape == (), f"expected scalar (), got {q.shape}"


def test_single_q_output_is_finite(single_q, single_q_params, batched_inputs):
    obs, action = batched_inputs
    q = single_q.apply(single_q_params, obs, action)
    assert jnp.all(jnp.isfinite(q))


def test_single_q_jit_compatible(single_q, single_q_params, batched_inputs):
    obs, action = batched_inputs
    q = jax.jit(single_q.apply)(single_q_params, obs, action)
    assert q.shape == (BATCH_SIZE,)


def test_single_q_gradient_flows(single_q, single_q_params, batched_inputs):
    obs, action = batched_inputs

    def loss(params):
        return jnp.mean(single_q.apply(params, obs, action))

    grads = jax.grad(loss)(single_q_params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(jnp.any(g != 0) for g in leaves), "some gradients are all-zero"


# ---------------------------------------------------------------------------
# DenseDoubleQNetworkContinuousAction
# ---------------------------------------------------------------------------

def test_double_q_returns_two_tensors(double_q, double_q_params, batched_inputs):
    obs, action = batched_inputs
    out = double_q.apply(double_q_params, obs, action)
    assert len(out) == 2, f"expected 2-tuple, got {len(out)} elements"


def test_double_q_batched_output_shapes(double_q, double_q_params, batched_inputs):
    obs, action = batched_inputs
    q1, q2 = double_q.apply(double_q_params, obs, action)
    assert q1.shape == (BATCH_SIZE,), f"q1: expected ({BATCH_SIZE},), got {q1.shape}"
    assert q2.shape == (BATCH_SIZE,), f"q2: expected ({BATCH_SIZE},), got {q2.shape}"


def test_double_q_unbatched_output_shapes(double_q, double_q_params, single_inputs):
    obs, action = single_inputs
    q1, q2 = double_q.apply(double_q_params, obs, action)
    assert q1.shape == (), f"q1: expected scalar (), got {q1.shape}"
    assert q2.shape == (), f"q2: expected scalar (), got {q2.shape}"


def test_double_q_networks_differ(double_q, double_q_params, batched_inputs):
    """The two Q-heads must have independent parameters and produce different outputs."""
    obs, action = batched_inputs
    q1, q2 = double_q.apply(double_q_params, obs, action)
    assert not jnp.allclose(q1, q2), "q1 and q2 are identical — networks may share weights"


def test_double_q_min_is_lower_bound(double_q, double_q_params, batched_inputs):
    obs, action = batched_inputs
    q1, q2 = double_q.apply(double_q_params, obs, action)
    q_min = jnp.minimum(q1, q2)
    assert jnp.all(q_min <= q1) and jnp.all(q_min <= q2)


def test_double_q_jit_compatible(double_q, double_q_params, batched_inputs):
    obs, action = batched_inputs
    q1, q2 = jax.jit(double_q.apply)(double_q_params, obs, action)
    assert q1.shape == (BATCH_SIZE,)
    assert q2.shape == (BATCH_SIZE,)


def test_double_q_gradient_flows(double_q, double_q_params, batched_inputs):
    """Use q1+q2 so gradients flow through both heads.
    jnp.minimum would zero-out the larger network's gradients everywhere."""
    obs, action = batched_inputs

    def loss(params):
        q1, q2 = double_q.apply(params, obs, action)
        return jnp.mean(q1) + jnp.mean(q2)

    grads = jax.grad(loss)(double_q_params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(jnp.any(g != 0) for g in leaves), "some gradients are all-zero"
