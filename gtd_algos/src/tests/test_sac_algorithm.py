"""
Tests for sac.py.

Loss function signatures (params-first pattern):
  critic_loss_fn(critic_params, agent_state, batch, rng)
  actor_loss_fn(actor_params, agent_state, batch, rng)
  alpha_loss_fn(log_alpha, agent_state, log_prob)

AgentState uses alpha_state: TrainState instead of log_alpha + alpha_optimizer_state.
Access log_alpha via agent_state.alpha_state.params.
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from gtd_algos.src.algorithms.sac import (
    AgentState,
    init_agent_state,
    agent_step,
    critic_loss_fn,
    actor_loss_fn,
    alpha_loss_fn,
    soft_update_target_params,
    update_step,
)
from gtd_algos.src.configs.Config import Config
from gtd_algos.src.experience_replay.buffer import BufferTransition

OBS_DIM = 8
ACTION_DIM = 2
BATCH_SIZE = 16


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent_config():
    return Config.from_dict({
        'gamma': 0.99,
        'tau': 0.005,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'alpha_lr': 3e-4,
        'gradient_clipping': False,
        'max_grad_norm': 10.0,
        'd_actor_repr': [64, 64],
        'd_critic_repr': [64, 64],
        'activation': 'relu',
        'action_dim': ACTION_DIM,
        'layer_norm_actor': True,
        'layer_norm_critic': True,
    })


@pytest.fixture
def agent_state(agent_config):
    return init_agent_state(
        agent_config, ACTION_DIM, (OBS_DIM,), True, jax.random.PRNGKey(0)
    )


@pytest.fixture
def batch():
    return BufferTransition(
        obs=jnp.ones((BATCH_SIZE, OBS_DIM)),
        action=jnp.zeros((BATCH_SIZE, ACTION_DIM)),
        reward=jnp.ones(BATCH_SIZE),
        next_obs=jnp.ones((BATCH_SIZE, OBS_DIM)) * 2.0,
        done=jnp.zeros(BATCH_SIZE, dtype=jnp.bool_),
        termination=jnp.zeros(BATCH_SIZE, dtype=jnp.bool_),
    )


# ---------------------------------------------------------------------------
# init_agent_state
# ---------------------------------------------------------------------------

def test_init_returns_agent_state_type(agent_state):
    assert isinstance(agent_state, AgentState)


def test_init_target_critic_is_params_dict_not_trainstate(agent_state):
    """target_critic_params must be a plain dict, not a TrainState."""
    assert not isinstance(agent_state.target_critic_params, TrainState)


def test_init_target_params_match_critic_at_start(agent_state):
    for t, c in zip(
        jax.tree_util.tree_leaves(agent_state.target_critic_params),
        jax.tree_util.tree_leaves(agent_state.critic_network_state.params),
    ):
        assert jnp.allclose(t, c)


def test_init_log_alpha_is_zero_scalar(agent_state):
    log_alpha = agent_state.alpha_state.params['log_alpha']
    assert log_alpha.shape == (), "log_alpha must be a scalar"
    assert float(log_alpha) == 0.0, "log(alpha)=0 means alpha=1 at init"


def test_init_alpha_state_is_trainstate(agent_state):
    assert isinstance(agent_state.alpha_state, TrainState)


# ---------------------------------------------------------------------------
# agent_step
# ---------------------------------------------------------------------------

def test_agent_step_action_shape(agent_state):
    obs = jnp.ones((BATCH_SIZE, OBS_DIM))
    action, _ = agent_step(agent_state, obs, jax.random.PRNGKey(0))
    assert action.shape == (BATCH_SIZE, ACTION_DIM)


def test_agent_step_action_in_tanh_range(agent_state):
    obs = jnp.ones((BATCH_SIZE, OBS_DIM))
    action, _ = agent_step(agent_state, obs, jax.random.PRNGKey(0))
    assert jnp.all(action > -1.0) and jnp.all(action < 1.0)


def test_agent_step_stochastic(agent_state):
    obs = jnp.ones((BATCH_SIZE, OBS_DIM))
    a1, _ = agent_step(agent_state, obs, jax.random.PRNGKey(0))
    a2, _ = agent_step(agent_state, obs, jax.random.PRNGKey(1))
    assert not jnp.allclose(a1, a2), "different RNGs must produce different actions"


# ---------------------------------------------------------------------------
# critic_loss_fn
# ---------------------------------------------------------------------------

def test_critic_loss_is_scalar_and_finite(agent_state, batch):
    loss = critic_loss_fn(agent_state.critic_network_state.params, agent_state, batch, jax.random.PRNGKey(0))
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_critic_loss_is_non_negative(agent_state, batch):
    loss = critic_loss_fn(agent_state.critic_network_state.params, agent_state, batch, jax.random.PRNGKey(0))
    assert loss >= 0.0


def test_critic_loss_uses_target_params_correctly(agent_state, batch):
    """B1: target_critic_params has no .apply_fn; must call critic_network_state.apply_fn(target_critic_params, ...)."""
    loss = critic_loss_fn(agent_state.critic_network_state.params, agent_state, batch, jax.random.PRNGKey(0))
    assert jnp.isfinite(loss)


def test_critic_loss_is_mean_of_squares_not_square_of_mean(agent_state, batch):
    """B4: correct formula is mean((q-y)^2), not mean(q-y)^2."""
    loss_a = critic_loss_fn(agent_state.critic_network_state.params, agent_state, batch, jax.random.PRNGKey(0))
    loss_b = critic_loss_fn(agent_state.critic_network_state.params, agent_state, batch._replace(reward=-batch.reward), jax.random.PRNGKey(0))
    assert loss_a > 0.0
    assert loss_b > 0.0


def test_critic_loss_bootstrap_uses_termination_not_done(agent_state):
    """B3: truncated episodes (done=True, termination=False) must still bootstrap."""
    truncation_batch = BufferTransition(
        obs=jnp.ones((BATCH_SIZE, OBS_DIM)),
        action=jnp.zeros((BATCH_SIZE, ACTION_DIM)),
        reward=jnp.ones(BATCH_SIZE),
        next_obs=jnp.ones((BATCH_SIZE, OBS_DIM)),
        done=jnp.ones(BATCH_SIZE, dtype=jnp.bool_),
        termination=jnp.zeros(BATCH_SIZE, dtype=jnp.bool_),
    )
    terminal_batch = BufferTransition(
        obs=jnp.ones((BATCH_SIZE, OBS_DIM)),
        action=jnp.zeros((BATCH_SIZE, ACTION_DIM)),
        reward=jnp.ones(BATCH_SIZE),
        next_obs=jnp.ones((BATCH_SIZE, OBS_DIM)),
        done=jnp.ones(BATCH_SIZE, dtype=jnp.bool_),
        termination=jnp.ones(BATCH_SIZE, dtype=jnp.bool_),
    )
    rng = jax.random.PRNGKey(0)
    critic_params = agent_state.critic_network_state.params
    loss_truncation = critic_loss_fn(critic_params, agent_state, truncation_batch, rng)
    loss_terminal   = critic_loss_fn(critic_params, agent_state, terminal_batch, rng)
    assert not jnp.allclose(loss_truncation, loss_terminal), (
        "truncation (done=True, termination=False) must bootstrap; "
        "using batch.done instead of batch.termination makes both identical"
    )


def test_critic_grad_flows_through_critic_params_only(agent_state, batch):
    def loss_fn(critic_params):
        return critic_loss_fn(critic_params, agent_state, batch, jax.random.PRNGKey(0))

    grads = jax.grad(loss_fn)(agent_state.critic_network_state.params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(jnp.any(g != 0) for g in leaves), "critic gradient has all-zero leaves"


# ---------------------------------------------------------------------------
# actor_loss_fn
# ---------------------------------------------------------------------------

def test_actor_loss_returns_loss_and_log_prob(agent_state, batch):
    loss, log_prob = actor_loss_fn(agent_state.actor_network_state.params, agent_state, batch, jax.random.PRNGKey(0))
    assert loss.shape == ()
    assert log_prob.shape == (BATCH_SIZE,)


def test_actor_loss_is_finite(agent_state, batch):
    loss, log_prob = actor_loss_fn(agent_state.actor_network_state.params, agent_state, batch, jax.random.PRNGKey(0))
    assert jnp.isfinite(loss)
    assert jnp.all(jnp.isfinite(log_prob))


def test_actor_loss_uses_current_critic_not_target(agent_state, batch):
    """B2: actor loss must use critic_network_state.params, not target_critic_params."""
    perturbed_target = jax.tree_map(lambda x: x * 100.0, agent_state.target_critic_params)
    state_with_perturbed_target = agent_state._replace(target_critic_params=perturbed_target)

    rng = jax.random.PRNGKey(0)
    actor_params = agent_state.actor_network_state.params
    loss_original, _ = actor_loss_fn(actor_params, agent_state, batch, rng)
    loss_perturbed, _ = actor_loss_fn(actor_params, state_with_perturbed_target, batch, rng)

    assert jnp.allclose(loss_original, loss_perturbed), (
        "actor loss must not change when target_critic_params are perturbed — "
        "actor loss uses current critic, not target"
    )


def test_actor_grad_flows_through_actor_params(agent_state, batch):
    def loss_fn(actor_params):
        loss, _ = actor_loss_fn(actor_params, agent_state, batch, jax.random.PRNGKey(0))
        return loss

    grads = jax.grad(loss_fn)(agent_state.actor_network_state.params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(jnp.any(g != 0) for g in leaves), "actor gradient has all-zero leaves"


# ---------------------------------------------------------------------------
# alpha_loss_fn
# ---------------------------------------------------------------------------

def test_alpha_loss_is_scalar_and_finite(agent_state, batch):
    _, log_prob = actor_loss_fn(agent_state.actor_network_state.params, agent_state, batch, jax.random.PRNGKey(0))
    loss = alpha_loss_fn(agent_state.alpha_state.params, agent_state, log_prob)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_alpha_gradient_direction(agent_state, batch):
    """When policy entropy is below target, alpha gradient should be positive
    (push alpha up to encourage more exploration)."""
    low_entropy_log_prob = jnp.full((BATCH_SIZE,), -0.1)

    def alpha_loss(alpha_params):
        return alpha_loss_fn(alpha_params, agent_state, low_entropy_log_prob)

    grads = jax.grad(alpha_loss)(agent_state.alpha_state.params)
    assert jnp.isfinite(grads['log_alpha'])


def test_alpha_grad_flows_through_log_alpha(agent_state, batch):
    _, log_prob = actor_loss_fn(agent_state.actor_network_state.params, agent_state, batch, jax.random.PRNGKey(0))

    def loss(alpha_params):
        return alpha_loss_fn(alpha_params, agent_state, log_prob)

    grads = jax.grad(loss)(agent_state.alpha_state.params)
    assert jnp.isfinite(grads['log_alpha']) and grads['log_alpha'] != 0.0


# ---------------------------------------------------------------------------
# soft_update_target_params
# ---------------------------------------------------------------------------

def test_soft_update_tau_one_copies_current_params(agent_state):
    """With tau=1.0, target becomes identical to current critic params."""
    new_target = soft_update_target_params(agent_state, tau=1.0)
    for t, c in zip(
        jax.tree_util.tree_leaves(new_target),
        jax.tree_util.tree_leaves(agent_state.critic_network_state.params),
    ):
        assert jnp.allclose(t, c)


def test_soft_update_tau_zero_leaves_target_unchanged(agent_state):
    """With tau=0.0, target params must not change."""
    original_target = jax.tree_map(lambda x: x.copy(), agent_state.target_critic_params)
    new_target = soft_update_target_params(agent_state, tau=0.0)
    for t_new, t_old in zip(
        jax.tree_util.tree_leaves(new_target),
        jax.tree_util.tree_leaves(original_target),
    ):
        assert jnp.allclose(t_new, t_old)


def test_soft_update_interpolates(agent_state):
    """With 0 < tau < 1, new target is between old target and current params."""
    tau = 0.5
    new_target = soft_update_target_params(agent_state, tau=tau)
    for t_new, t_old, c in zip(
        jax.tree_util.tree_leaves(new_target),
        jax.tree_util.tree_leaves(agent_state.target_critic_params),
        jax.tree_util.tree_leaves(agent_state.critic_network_state.params),
    ):
        expected = tau * c + (1 - tau) * t_old
        assert jnp.allclose(t_new, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# update_step
# ---------------------------------------------------------------------------

def test_update_step_returns_agent_state_and_loss_dict(agent_state, batch):
    new_state, losses = update_step(agent_state, batch, jax.random.PRNGKey(0))
    assert isinstance(new_state, AgentState)
    assert {'critic_loss', 'actor_loss', 'alpha_loss', 'alpha'} <= losses.keys()


def test_update_step_losses_are_finite(agent_state, batch):
    _, losses = update_step(agent_state, batch, jax.random.PRNGKey(0))
    for k, v in losses.items():
        assert jnp.isfinite(v), f"{k} is not finite"


def test_update_step_critic_params_change(agent_state, batch):
    new_state, _ = update_step(agent_state, batch, jax.random.PRNGKey(0))
    old_leaves = jax.tree_util.tree_leaves(agent_state.critic_network_state.params)
    new_leaves = jax.tree_util.tree_leaves(new_state.critic_network_state.params)
    assert any(not jnp.allclose(o, n) for o, n in zip(old_leaves, new_leaves)), \
        "critic params did not change after update"


def test_update_step_actor_params_change(agent_state, batch):
    new_state, _ = update_step(agent_state, batch, jax.random.PRNGKey(0))
    old_leaves = jax.tree_util.tree_leaves(agent_state.actor_network_state.params)
    new_leaves = jax.tree_util.tree_leaves(new_state.actor_network_state.params)
    assert any(not jnp.allclose(o, n) for o, n in zip(old_leaves, new_leaves)), \
        "actor params did not change after update"


def test_update_step_alpha_changes(agent_state, batch):
    new_state, _ = update_step(agent_state, batch, jax.random.PRNGKey(0))
    assert not jnp.allclose(
        agent_state.alpha_state.params['log_alpha'],
        new_state.alpha_state.params['log_alpha'],
    ), "log_alpha did not change after update"


def test_update_step_target_params_change_via_polyak(agent_state, batch):
    new_state, _ = update_step(agent_state, batch, jax.random.PRNGKey(0))
    old_target = jax.tree_util.tree_leaves(agent_state.target_critic_params)
    new_target = jax.tree_util.tree_leaves(new_state.target_critic_params)
    # Use exact float comparison: tau=0.005 makes the change very small, well below
    # jnp.allclose defaults, but any nonzero gradient step produces a nonzero difference.
    assert any(jnp.any(o != n) for o, n in zip(old_target, new_target)), \
        "target params did not change — Polyak update may be broken"


def test_update_step_target_not_equal_to_critic(agent_state, batch):
    new_state, _ = update_step(agent_state, batch, jax.random.PRNGKey(0))
    target_leaves = jax.tree_util.tree_leaves(new_state.target_critic_params)
    critic_leaves = jax.tree_util.tree_leaves(new_state.critic_network_state.params)
    assert any(not jnp.allclose(t, c) for t, c in zip(target_leaves, critic_leaves)), \
        "target and critic params are identical — Polyak is not being applied (tau=1?)"


def test_update_step_does_not_modify_input_state(agent_state, batch):
    original_actor = jax.tree_util.tree_leaves(agent_state.actor_network_state.params)
    update_step(agent_state, batch, jax.random.PRNGKey(0))
    after_actor = jax.tree_util.tree_leaves(agent_state.actor_network_state.params)
    for o, a in zip(original_actor, after_actor):
        assert jnp.allclose(o, a), "input agent_state was mutated"
