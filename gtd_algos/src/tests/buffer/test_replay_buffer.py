import jax
import jax.numpy as jnp
import pytest

from gtd_algos.src.experience_replay.buffer import (
    BufferTransition,
    ReplayBufferState,
    init_buffer,
    add_transition,
    sample_transitions,
)

CAPACITY = 10
OBS_SHAPE = (4,)
ACTION_DIM = 2


@pytest.fixture
def empty_buffer():
    return init_buffer(capacity=CAPACITY, obs_shape=OBS_SHAPE, action_dim=ACTION_DIM)


@pytest.fixture
def dummy_transition():
    return BufferTransition(
        obs=jnp.ones(OBS_SHAPE),
        action=jnp.ones(ACTION_DIM),
        reward=jnp.array(1.0),
        next_obs=jnp.ones(OBS_SHAPE) * 2,
        done=jnp.array(False),
        termination=jnp.array(False),
    )


def test_init_buffer_is_empty(empty_buffer):
    assert empty_buffer.size == 0
    assert empty_buffer.idx == 0
    assert empty_buffer.obs.shape == (CAPACITY, *OBS_SHAPE)
    assert empty_buffer.action.shape == (CAPACITY, ACTION_DIM)


def test_add_single_transition(empty_buffer, dummy_transition):
    buf = add_transition(empty_buffer, dummy_transition)
    assert buf.size == 1
    assert buf.idx == 1
    assert jnp.allclose(buf.obs[0], dummy_transition.obs)
    assert jnp.allclose(buf.next_obs[0], dummy_transition.next_obs)


def test_size_caps_at_capacity(empty_buffer, dummy_transition):
    buf = empty_buffer
    for _ in range(CAPACITY + 1):
        buf = add_transition(buf, dummy_transition)
    assert buf.size == CAPACITY


def test_circular_wraparound(empty_buffer, dummy_transition):
    buf = empty_buffer
    for _ in range(CAPACITY + 1):
        buf = add_transition(buf, dummy_transition)
    # After CAPACITY+1 writes, idx should have wrapped to 1
    assert buf.idx == 1


def test_sample_batch_shape(empty_buffer, dummy_transition):
    buf = empty_buffer
    for _ in range(CAPACITY):
        buf = add_transition(buf, dummy_transition)

    rng = jax.random.PRNGKey(0)
    batch_size = 4
    batch = sample_transitions(buf, batch_size=batch_size, rng=rng)

    assert batch.obs.shape == (batch_size, *OBS_SHAPE)
    assert batch.action.shape == (batch_size, ACTION_DIM)
    assert batch.reward.shape == (batch_size,)
    assert batch.next_obs.shape == (batch_size, *OBS_SHAPE)


def test_sample_only_from_filled_slots():
    buf = init_buffer(capacity=100, obs_shape=(4,), action_dim=2)
    t = BufferTransition(
        obs=jnp.array([1.0, 2.0, 3.0, 4.0]),
        action=jnp.ones(2),
        reward=jnp.array(0.5),
        next_obs=jnp.zeros(4),
        done=jnp.array(False),
        termination=jnp.array(False),
    )
    # Only add 5 transitions into a capacity-100 buffer
    for _ in range(5):
        buf = add_transition(buf, t)

    rng = jax.random.PRNGKey(42)
    batch = sample_transitions(buf, batch_size=5, rng=rng)
    # All sampled obs should match what we stored, not the zero-initialised slots
    assert jnp.all(batch.obs[:, 0] == 1.0)


def test_add_transition_is_jit_compatible(empty_buffer, dummy_transition):
    add_jit = jax.jit(add_transition)
    buf = add_jit(empty_buffer, dummy_transition)
    buf = add_jit(buf, dummy_transition)  # second call must not retrace
    assert buf.size == 2
