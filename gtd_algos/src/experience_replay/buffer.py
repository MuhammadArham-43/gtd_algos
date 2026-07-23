from typing import NamedTuple
import functools
import jax
import jax.numpy as jnp
import numpy as np

class BufferTransition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    done: jnp.ndarray
    termination: jnp.ndarray

class ReplayBufferState(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    done: jnp.ndarray
    termination: jnp.ndarray
    idx: jnp.ndarray
    size: jnp.ndarray

def init_buffer(capacity: int, obs_shape: tuple, action_dim: int) -> ReplayBufferState:
    return ReplayBufferState(
        obs=jnp.zeros((capacity, *obs_shape), dtype=jnp.float32),
        action=jnp.zeros((capacity, action_dim), dtype=jnp.float32),
        reward=jnp.zeros((capacity,), dtype=jnp.float32),
        next_obs=jnp.zeros((capacity, *obs_shape), dtype=jnp.float32),
        done=jnp.zeros((capacity,), dtype=jnp.float32),
        termination=jnp.zeros((capacity,), dtype=jnp.float32),
        idx=jnp.array(0, dtype=jnp.int32),
        size=jnp.array(0, dtype=jnp.int32)
    )

@functools.partial(jax.jit, donate_argnums=(0,))
def add_transition(buffer_state: ReplayBufferState, transition: BufferTransition) -> ReplayBufferState:
    capacity = buffer_state.obs.shape[0]
    idx = buffer_state.idx
    buffer_state = buffer_state._replace(
        obs=buffer_state.obs.at[idx].set(transition.obs),
        action=buffer_state.action.at[idx].set(transition.action),
        reward=buffer_state.reward.at[idx].set(transition.reward),
        next_obs=buffer_state.next_obs.at[idx].set(transition.next_obs),
        done=buffer_state.done.at[idx].set(transition.done),
        termination=buffer_state.termination.at[idx].set(transition.termination),
        idx=(idx + 1) % capacity,
        size=jnp.minimum(buffer_state.size + 1, capacity)
    )
    return buffer_state

@functools.partial(jax.jit, static_argnums=(1,))
def sample_transitions(buffer_state: ReplayBufferState, batch_size: int, rng: jnp.ndarray) -> BufferTransition:
    idxs = jax.random.randint(rng, (batch_size,), 0, buffer_state.size)
    return BufferTransition(
        obs=buffer_state.obs[idxs],
        action=buffer_state.action[idxs],
        reward=buffer_state.reward[idxs],
        next_obs=buffer_state.next_obs[idxs],
        done=buffer_state.done[idxs],
        termination=buffer_state.termination[idxs]
    )
