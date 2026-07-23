import jax
import jax.numpy as jnp
import pytest
from flax.linen.initializers import orthogonal, constant

from gtd_algos.src.agents.ActorCritic import SACContinousActor

OBS_DIM = 17       # e.g. HalfCheetah-v4
ACTION_DIM = 6
HIDDENS = (256, 256)
BATCH_SIZE = 4
RNG = jax.random.PRNGKey(0)


@pytest.fixture
def actor():
    return SACContinousActor(
        action_dim=ACTION_DIM,
        d_actor=HIDDENS,
        activation="relu",
    )


@pytest.fixture
def batched_obs():
    return jnp.ones((BATCH_SIZE, OBS_DIM))


@pytest.fixture
def single_obs():
    return jnp.ones(OBS_DIM)


@pytest.fixture
def actor_params(actor, batched_obs):
    rng, sample_rng = jax.random.split(RNG)
    # init must be done in stochastic mode so both Dense heads are registered
    return actor.init(rng, batched_obs, sample_rng, False)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_init_in_stochastic_mode_registers_both_heads(actor, batched_obs):
    """Both mean and log_std Dense layers must exist after init."""
    rng, sample_rng = jax.random.split(RNG)
    params = actor.init(rng, batched_obs, sample_rng, False)
    param_leaves = jax.tree_util.tree_leaves(params)
    assert len(param_leaves) > 0


def test_init_in_deterministic_mode_still_has_log_std_params(actor, batched_obs):
    """Bug 2: if only deterministic init is run, log_std Dense is not registered.
    This test will fail until the branching is fixed."""
    rng, sample_rng = jax.random.split(RNG)
    params = actor.init(rng, batched_obs, None, True)
    # stochastic forward pass must work with params obtained from deterministic init
    _, sample_rng = jax.random.split(rng)
    action, log_prob = actor.apply(params, batched_obs, sample_rng, False)
    assert action.shape == (BATCH_SIZE, ACTION_DIM)


# ---------------------------------------------------------------------------
# Output shapes — stochastic
# ---------------------------------------------------------------------------

def test_stochastic_action_shape_batched(actor, actor_params, batched_obs):
    _, sample_rng = jax.random.split(RNG)
    action, log_prob = actor.apply(actor_params, batched_obs, sample_rng, False)
    assert action.shape == (BATCH_SIZE, ACTION_DIM)


def test_stochastic_log_prob_shape_batched(actor, actor_params, batched_obs):
    _, sample_rng = jax.random.split(RNG)
    action, log_prob = actor.apply(actor_params, batched_obs, sample_rng, False)
    assert log_prob.shape == (BATCH_SIZE,)


def test_stochastic_action_shape_single(actor, actor_params, single_obs):
    _, sample_rng = jax.random.split(RNG)
    action, log_prob = actor.apply(actor_params, single_obs, sample_rng, False)
    assert action.shape == (ACTION_DIM,)
    assert log_prob.shape == ()


# ---------------------------------------------------------------------------
# Output shapes — deterministic
# ---------------------------------------------------------------------------

def test_deterministic_action_shape(actor, actor_params, batched_obs):
    action, log_prob = actor.apply(actor_params, batched_obs, None, True)
    assert action.shape == (BATCH_SIZE, ACTION_DIM)
    assert log_prob is None


# ---------------------------------------------------------------------------
# Action bounds
# ---------------------------------------------------------------------------

def test_actions_are_in_tanh_range(actor, actor_params, batched_obs):
    _, sample_rng = jax.random.split(RNG)
    action, _ = actor.apply(actor_params, batched_obs, sample_rng, False)
    assert jnp.all(action > -1.0) and jnp.all(action < 1.0)


def test_deterministic_action_in_tanh_range(actor, actor_params, batched_obs):
    action, _ = actor.apply(actor_params, batched_obs, None, True)
    assert jnp.all(action > -1.0) and jnp.all(action < 1.0)


# ---------------------------------------------------------------------------
# Log-prob validity
# ---------------------------------------------------------------------------

def test_log_prob_is_finite(actor, actor_params, batched_obs):
    _, sample_rng = jax.random.split(RNG)
    _, log_prob = actor.apply(actor_params, batched_obs, sample_rng, False)
    assert jnp.all(jnp.isfinite(log_prob))


def test_log_prob_is_negative(actor, actor_params, batched_obs):
    """Log-prob of a tanh-Gaussian over a bounded support should be <= 0
    for typical actions. Not guaranteed for all inputs, but holds for action_dim >= 1
    under a broad Gaussian."""
    _, sample_rng = jax.random.split(RNG)
    _, log_prob = actor.apply(actor_params, batched_obs, sample_rng, False)
    assert jnp.mean(log_prob) < 0.0


# ---------------------------------------------------------------------------
# Stochasticity
# ---------------------------------------------------------------------------

def test_different_rngs_give_different_actions(actor, actor_params, batched_obs):
    rng1, rng2 = jax.random.split(RNG)
    a1, _ = actor.apply(actor_params, batched_obs, rng1, False)
    a2, _ = actor.apply(actor_params, batched_obs, rng2, False)
    assert not jnp.allclose(a1, a2)


def test_deterministic_is_reproducible(actor, actor_params, batched_obs):
    a1, _ = actor.apply(actor_params, batched_obs, None, True)
    a2, _ = actor.apply(actor_params, batched_obs, None, True)
    assert jnp.allclose(a1, a2)


# ---------------------------------------------------------------------------
# Gradient flow (needed for actor update in SAC)
# ---------------------------------------------------------------------------

def test_gradient_flows_through_log_prob(actor, actor_params, batched_obs):
    sample_rng = jax.random.PRNGKey(1)

    def actor_loss(params):
        _, log_prob = actor.apply(params, batched_obs, sample_rng, False)
        return jnp.mean(log_prob)

    grads = jax.grad(actor_loss)(actor_params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(jnp.any(g != 0) for g in leaves), "some param gradients are all-zero"


def test_gradient_flows_through_action(actor, actor_params, batched_obs):
    """Actions must be differentiable w.r.t. params for the Q-based actor loss."""
    sample_rng = jax.random.PRNGKey(2)

    def actor_loss(params):
        action, _ = actor.apply(params, batched_obs, sample_rng, False)
        return jnp.mean(action)

    grads = jax.grad(actor_loss)(actor_params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(jnp.any(g != 0) for g in leaves), "some param gradients are all-zero"


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------

def test_stochastic_forward_is_jit_compatible(actor, actor_params, batched_obs):
    sample_rng = jax.random.PRNGKey(3)
    apply_jit = jax.jit(actor.apply, static_argnums=(3,))
    action, log_prob = apply_jit(actor_params, batched_obs, sample_rng, False)
    assert action.shape == (BATCH_SIZE, ACTION_DIM)


def test_deterministic_forward_is_jit_compatible(actor, actor_params, batched_obs):
    apply_jit = jax.jit(actor.apply, static_argnums=(3,))
    action, _ = apply_jit(actor_params, batched_obs, None, True)
    assert action.shape == (BATCH_SIZE, ACTION_DIM)
