"""
Tests for the SAC training loop in sac_main.py.

Strategy: mock the agent and environment to isolate loop logic from network
correctness. Each test targets one specific behaviour or bug.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gtd_algos.src.algorithms.agent import Agent
from gtd_algos.src.experience_replay.buffer import init_buffer
from gtd_algos.src.experiments.gym_exps.sac_main import exp_step

OBS_DIM = 4
ACTION_DIM = 2
CAPACITY = 50
WARMUP = 5
BATCH_SIZE = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockEnv:
    """Minimal gymnasium-compatible environment."""

    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM, episode_len=1000):
        self.observation_space = SimpleNamespace(shape=(obs_dim,))
        self.action_space = SimpleNamespace(shape=(action_dim,))
        self._obs_dim = obs_dim
        self._episode_len = episode_len
        self._step_count = 0
        self.received_actions = []

    def reset(self):
        self._step_count = 0
        return np.zeros(self._obs_dim, dtype=np.float32), {}

    def step(self, action):
        self.received_actions.append(np.asarray(action))
        self._step_count += 1
        obs = np.random.randn(self._obs_dim).astype(np.float32)
        reward = float(np.random.randn())
        terminated = self._step_count >= self._episode_len
        info = {'episode': {'r': 1.0, 'l': self._step_count}} if terminated else {}
        return obs, reward, terminated, False, info


def make_agent_state(warmup_steps=WARMUP, batch_size=BATCH_SIZE, update_steps=1):
    return SimpleNamespace(
        agent_config=SimpleNamespace(
            warmup_steps=warmup_steps,
            batch_size=batch_size,
            update_steps=update_steps,
        )
    )


def make_agent(update_fn=None):
    if update_fn is None:
        update_fn = lambda state, batch, rng: (state, {})

    def step_fn(agent_state, obs, rng):
        return jnp.zeros((1, ACTION_DIM))   # batched — (1, action_dim)

    return Agent(init_state=None, step=step_fn, update=update_fn)


@pytest.fixture
def env():
    return MockEnv()


@pytest.fixture
def agent():
    return make_agent()


@pytest.fixture
def buf():
    return init_buffer(capacity=CAPACITY, obs_shape=(OBS_DIM,), action_dim=ACTION_DIM)


@pytest.fixture
def runner(buf):
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    rng = jax.random.PRNGKey(0)
    return (make_agent_state(), buf, obs, rng)


# ---------------------------------------------------------------------------
# Bug 2 — action shape
# ---------------------------------------------------------------------------

def test_action_stored_in_buffer_as_1d(runner, env, agent):
    """actor returns (1, action_dim); buffer must store (action_dim,)."""
    new_runner, _ = exp_step(runner, env, idx=0, agent=agent)
    _, buffer_state, _, _ = new_runner
    assert buffer_state.action[0].shape == (ACTION_DIM,)


def test_env_receives_1d_action(runner, agent):
    """env.step must receive a 1-D action array, not (1, action_dim)."""
    env = MockEnv()
    exp_step(runner, env, idx=0, agent=agent)
    assert env.received_actions[0].ndim == 1
    assert env.received_actions[0].shape == (ACTION_DIM,)


# ---------------------------------------------------------------------------
# Buffer accumulation
# ---------------------------------------------------------------------------

def test_buffer_grows_one_per_step(runner, env, agent):
    state = runner
    for i in range(7):
        state, _ = exp_step(state, env, idx=i, agent=agent)
    assert state[1].size == 7


def test_buffer_caps_at_capacity(env, agent):
    small_cap = 5
    buf = init_buffer(capacity=small_cap, obs_shape=(OBS_DIM,), action_dim=ACTION_DIM)
    state = (make_agent_state(warmup_steps=9999), buf,
             np.zeros(OBS_DIM, dtype=np.float32), jax.random.PRNGKey(0))
    for i in range(small_cap + 3):
        state, _ = exp_step(state, env, idx=i, agent=agent)
    assert state[1].size == small_cap


# ---------------------------------------------------------------------------
# Warmup / update gating
# ---------------------------------------------------------------------------

def test_no_update_before_warmup(env):
    update_mock = MagicMock(side_effect=lambda s, b, r: (s, {}))
    agent = make_agent(update_fn=update_mock)
    state = (make_agent_state(warmup_steps=10),
             init_buffer(CAPACITY, (OBS_DIM,), ACTION_DIM),
             np.zeros(OBS_DIM, dtype=np.float32), jax.random.PRNGKey(0))

    for i in range(9):   # idx 0..8, all strictly < warmup_steps=10
        state, _ = exp_step(state, env, idx=i, agent=agent)

    update_mock.assert_not_called()


def test_update_called_after_warmup_and_buffer_full(env):
    update_mock = MagicMock(side_effect=lambda s, b, r: (s, {}))
    agent = make_agent(update_fn=update_mock)
    state = (make_agent_state(warmup_steps=WARMUP, batch_size=BATCH_SIZE, update_steps=1),
             init_buffer(CAPACITY, (OBS_DIM,), ACTION_DIM),
             np.zeros(OBS_DIM, dtype=np.float32), jax.random.PRNGKey(0))

    for i in range(WARMUP + BATCH_SIZE + 2):
        state, _ = exp_step(state, env, idx=i, agent=agent)

    assert update_mock.call_count > 0


def test_no_update_when_buffer_smaller_than_batch(env):
    """Even past warmup, skip update when buffer.size < batch_size."""
    update_mock = MagicMock(side_effect=lambda s, b, r: (s, {}))
    agent = make_agent(update_fn=update_mock)
    batch_size = 20
    state = (make_agent_state(warmup_steps=2, batch_size=batch_size),
             init_buffer(CAPACITY, (OBS_DIM,), ACTION_DIM),
             np.zeros(OBS_DIM, dtype=np.float32), jax.random.PRNGKey(0))

    # Past warmup but only 5 transitions — still below batch_size=20
    for i in range(7):
        state, _ = exp_step(state, env, idx=i, agent=agent)

    update_mock.assert_not_called()


def test_update_steps_respected(env):
    """update_steps=3 must call agent.update exactly 3 times per eligible step."""
    update_mock = MagicMock(side_effect=lambda s, b, r: (s, {}))
    agent = make_agent(update_fn=update_mock)
    update_steps = 3
    state = (make_agent_state(warmup_steps=0, batch_size=BATCH_SIZE, update_steps=update_steps),
             init_buffer(CAPACITY, (OBS_DIM,), ACTION_DIM),
             np.zeros(OBS_DIM, dtype=np.float32), jax.random.PRNGKey(0))

    # Run exactly BATCH_SIZE steps so the buffer is full enough from step BATCH_SIZE onward
    n = BATCH_SIZE + 1
    for i in range(n):
        state, _ = exp_step(state, env, idx=i, agent=agent)

    # Only steps where buffer.size >= BATCH_SIZE trigger updates
    eligible_steps = n - BATCH_SIZE + 1   # steps BATCH_SIZE..n inclusive
    assert update_mock.call_count == eligible_steps * update_steps


# ---------------------------------------------------------------------------
# Bug 1 — RNG handling
# ---------------------------------------------------------------------------

def test_rng_advances_each_step(runner, env, agent):
    """RNG in runner_state must differ after every call."""
    state = runner
    prev_rng = np.array(state[3])
    for i in range(5):
        state, _ = exp_step(state, env, idx=i, agent=agent)
        cur_rng = np.array(state[3])
        assert not np.array_equal(prev_rng, cur_rng), f"RNG unchanged at step {i}"
        prev_rng = cur_rng


def test_sample_and_update_use_different_rng_keys(env):
    """Bug 1: sample_transitions and agent.update must receive distinct keys."""
    sample_keys, update_keys = [], []

    import gtd_algos.src.experience_replay.buffer as buf_module
    original_sample = buf_module.sample_transitions

    def capturing_sample(buf, batch_size, rng):
        sample_keys.append(np.array(rng))
        return original_sample(buf, batch_size, rng)

    def capturing_update(agent_state, batch, rng):
        update_keys.append(np.array(rng))
        return agent_state, {}

    agent = make_agent(update_fn=capturing_update)
    state = (make_agent_state(warmup_steps=WARMUP, batch_size=BATCH_SIZE, update_steps=1),
             init_buffer(CAPACITY, (OBS_DIM,), ACTION_DIM),
             np.zeros(OBS_DIM, dtype=np.float32), jax.random.PRNGKey(0))

    with patch('gtd_algos.src.experiments.gym_exps.sac_main.sample_transitions',
               side_effect=capturing_sample):
        for i in range(WARMUP + BATCH_SIZE + 5):
            state, _ = exp_step(state, env, idx=i, agent=agent)

    assert len(sample_keys) > 0, "sample_transitions was never called"
    assert len(update_keys) > 0, "agent.update was never called"
    for s, u in zip(sample_keys, update_keys):
        assert not np.array_equal(s, u), "same RNG key passed to sample and update"


# ---------------------------------------------------------------------------
# Episode reset
# ---------------------------------------------------------------------------

def test_obs_replaced_with_reset_obs_after_done(agent):
    """When terminated, the runner_state obs for the next step must be the reset obs."""
    reset_obs = np.full(OBS_DIM, 42.0, dtype=np.float32)

    terminal_env = MockEnv(episode_len=2)
    terminal_env.reset = lambda: (reset_obs.copy(), {})

    state = (make_agent_state(warmup_steps=9999),
             init_buffer(CAPACITY, (OBS_DIM,), ACTION_DIM),
             np.zeros(OBS_DIM, dtype=np.float32),
             jax.random.PRNGKey(0))

    # Step until termination
    for i in range(2):
        state, _ = exp_step(state, terminal_env, idx=i, agent=agent)

    _, _, obs_after_reset, _ = state
    assert np.allclose(obs_after_reset, reset_obs), \
        "obs after episode end must come from env.reset()"


# ---------------------------------------------------------------------------
# Bug 3 — result initialisation
# ---------------------------------------------------------------------------

def test_experiment_result_not_unbound_when_zero_steps():
    """Bug 3: `result` must be initialised before the loop.
    With total_steps=0 the loop never runs, so an uninitialised result raises
    UnboundLocalError."""
    from gtd_algos.src.experiments.gym_exps.sac_main import experiment

    mock_env = MockEnv()
    mock_env.observation_space = SimpleNamespace(shape=(OBS_DIM,))
    mock_env.action_space = SimpleNamespace(shape=(ACTION_DIM,))

    agent_config = SimpleNamespace(
        total_steps=0, gamma=0.99, buffer_capacity=100,
        warmup_steps=5, batch_size=4, update_steps=1,
    )
    env_config = SimpleNamespace(continuous_action=True)
    config = SimpleNamespace(agent_config=agent_config, env_config=env_config, exp_seed=0)

    def mock_init_state(agent_config, action_dim, obs_shape, continuous, rng):
        return make_agent_state(warmup_steps=5, batch_size=4, update_steps=1)

    agent = Agent(
        init_state=mock_init_state,
        step=lambda s, o, r: jnp.zeros((1, ACTION_DIM)),
        update=lambda s, b, r: (s, {}),
    )

    with patch('gtd_algos.src.experiments.gym_exps.sac_main.make_env',
               return_value=mock_env):
        out = experiment(config, agent)

    assert 'result' in out  # no UnboundLocalError
