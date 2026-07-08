from typing import NamedTuple, Any

import numpy as np
import jax
import jax.numpy as jnp
import optax
import wandb
from flax import struct
from flax.training.train_state import TrainState

from gtd_algos.src.configs.Config import Config
from gtd_algos.src.algorithms.agent import Agent
from gtd_algos.src.agents.ActorCritic import SACContinousActor
from gtd_algos.src.agents.value_networks import DenseDoubleQNetworkContinuousAction
from gtd_algos.src.experience_replay.buffer import BufferTransition


PRNGKey = Any

class AgentState(NamedTuple):
    agent_config: Config
    actor_network_state: TrainState
    critic_network_state: TrainState
    alpha_state: TrainState
    target_critic_params: Any

def init_agent_state(agent_config: Config, action_dim: int, obs_shape: tuple, continous_action: bool, rng: PRNGKey):
    agent_config = Config.from_dict({**agent_config.d, 'action_dim': action_dim})
    actor_network = SACContinousActor(
        action_dim=action_dim,
        d_actor=agent_config.d_actor_repr,
        activation=agent_config.activation,
        layer_norm=agent_config.layer_norm_actor,
    )
    critic_network = DenseDoubleQNetworkContinuousAction(
        layer_norm=agent_config.layer_norm_critic,
        hiddens=agent_config.d_critic_repr,
    )

    init_x = jnp.zeros((1, *obs_shape))
    init_a = jnp.zeros((1, action_dim))
    rng, _rng = jax.random.split(rng)
    actor_network_params = actor_network.init(rng, init_x, _rng)
    rng, _rng = jax.random.split(rng)
    critic_network_params = critic_network.init(_rng, init_x, init_a)
    
    def params_count(params):
        return sum(p.size for p in jax.tree_util.tree_leaves(params))

    print(
        f"Total Number of Parameters in Actor Network: {params_count(actor_network_params)}\n"
        f"Total Number of Parameters in Critic Network: {params_count(critic_network_params)}\n"
    )

    def new_optimizer(lr):
        adam = optax.adam(lr, eps=1e-5)
        if agent_config.gradient_clipping:
            return optax.chain(
                optax.clip_by_global_norm(agent_config.max_grad_norm),
                adam
            )
        return adam
    
    actor_network_state = TrainState.create(
        apply_fn=actor_network.apply,
        params=actor_network_params,
        tx=new_optimizer(agent_config.actor_lr)
    )
    critic_network_state = TrainState.create(
        apply_fn=critic_network.apply,
        params=critic_network_params,
        tx=new_optimizer(agent_config.critic_lr)
    )
    alpha_state = TrainState.create(
        apply_fn=lambda p: jnp.exp(p['log_alpha']),
        params={'log_alpha': jnp.array(0.0)},
        tx=new_optimizer(agent_config.alpha_lr)
    )

    return AgentState(
        agent_config=agent_config,
        actor_network_state=actor_network_state,
        critic_network_state=critic_network_state,
        alpha_state=alpha_state,
        target_critic_params=critic_network_params,
    )


@jax.jit
def agent_step(agent_state: AgentState, obs: jnp.ndarray, rng: PRNGKey) :
    action, log_prob = agent_state.actor_network_state.apply_fn(agent_state.actor_network_state.params, obs, rng)
    return action

def critic_loss_fn(critic_params, agent_state: AgentState, batch: BufferTransition, rng: PRNGKey):
    obs, action, reward, next_obs, done, termination = batch
    gamma = agent_state.agent_config.gamma
    alpha = jnp.exp(agent_state.alpha_state.params['log_alpha'])


    rng, _rng = jax.random.split(rng)
    next_a, next_a_logprob = agent_state.actor_network_state.apply_fn(agent_state.actor_network_state.params, next_obs, _rng)
    q1_t, q2_t = agent_state.critic_network_state.apply_fn(agent_state.target_critic_params, next_obs, next_a)
    next_v = jnp.minimum(q1_t, q2_t) - alpha * next_a_logprob
    y = batch.reward + gamma * (1.0 - batch.termination) * next_v
    y = jax.lax.stop_gradient(y)
    q1, q2 = agent_state.critic_network_state.apply_fn(critic_params, obs, action)
    return jnp.mean((q1 - y) ** 2 + (q2 - y) ** 2)

def actor_loss_fn(actor_params, agent_state: AgentState, batch: BufferTransition, rng: PRNGKey):
    alpha = jnp.exp(agent_state.alpha_state.params['log_alpha'])
    rng, _rng = jax.random.split(rng)
    action, log_prob = agent_state.actor_network_state.apply_fn(actor_params, batch.obs, _rng, deterministic=False)
    q1, q2 = agent_state.critic_network_state.apply_fn(agent_state.critic_network_state.params, batch.obs, action)
    q = jnp.minimum(q1, q2)
    return jnp.mean(alpha * log_prob - q), log_prob

def alpha_loss_fn(alpha_params, agent_state: AgentState, log_prob: jnp.ndarray):
    log_alpha = alpha_params['log_alpha']
    target_entropy = -agent_state.agent_config.action_dim
    alpha = jnp.exp(log_alpha)
    return jnp.mean(-alpha * (jax.lax.stop_gradient(log_prob) + target_entropy))

def soft_update_target_params(agent_state: AgentState, tau: float):
    new_target_params = jax.tree_util.tree_map(
        lambda t, s: tau * s + (1 - tau) * t,
        agent_state.target_critic_params,
        agent_state.critic_network_state.params
    )
    return new_target_params

@jax.jit
def update_step(agent_state: AgentState, batch: BufferTransition, rng: PRNGKey):
    critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(agent_state.critic_network_state.params, agent_state, batch, rng)
    critic_state = agent_state.critic_network_state.apply_gradients(grads=critic_grads)

    (actor_loss, log_prob), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(agent_state.actor_network_state.params, agent_state, batch, rng)
    actor_state = agent_state.actor_network_state.apply_gradients(grads=actor_grads)

    alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(agent_state.alpha_state.params, agent_state, log_prob)
    alpha_state = agent_state.alpha_state.apply_gradients(grads=alpha_grads)

    new_target_params = soft_update_target_params(agent_state, agent_state.agent_config.tau)

    agent_state = agent_state._replace(
        actor_network_state=actor_state,
        critic_network_state=critic_state,
        alpha_state=alpha_state,
        target_critic_params=new_target_params,
    )
    return agent_state, {
        "critic_loss": critic_loss,
        "actor_loss": actor_loss,
        "alpha_loss": alpha_loss,
        "alpha": jnp.exp(agent_state.alpha_state.params['log_alpha'])
    }

def wandb_sac_logging(result):
    loss_info = result.get('loss_info', {})
    episode_info = result.get('episode_info', {})
    env_steps = result.get('env_steps', 0)

    log_dict = {'env_steps': env_steps}

    if loss_info:
        log_dict.update({
            'critic_loss': float(loss_info['critic_loss']),
            'actor_loss':  float(loss_info['actor_loss']),
            'alpha_loss':  float(loss_info['alpha_loss']),
            'alpha':       float(loss_info['alpha']),
        })

    if 'episode' in episode_info:
        log_dict['undiscounted_return'] = float(episode_info['episode']['r'])
        log_dict['episode_length']      = int(episode_info['episode']['l'])

    wandb.log(log_dict)

SACAgent = Agent(init_agent_state, agent_step, update_step)