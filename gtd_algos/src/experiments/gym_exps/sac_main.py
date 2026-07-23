from functools import partial

import jax
import jax.numpy as jnp
import wandb

from gtd_algos.src.algorithms.agent import Agent
from gtd_algos.src.configs.ExpConfig import ExpConfig
from gtd_algos.src.algorithms.sac import SACAgent, wandb_sac_logging
from gtd_algos.src.envs.make_gym_envs import make_env
from gtd_algos.src.experiments.main import main

from gtd_algos.src.experience_replay.buffer import BufferTransition, ReplayBufferState, init_buffer, add_transition, sample_transitions


@partial(jax.jit, static_argnums=(0,))
def post_env_step(
    agent: Agent,
    agent_state,
    buffer_state,
    obs,
    action,
    reward,
    next_obs,
    done,
    terminated,
    rng,
    idx,
):
    t = BufferTransition(
        obs=obs,
        action=action,
        reward=reward,
        next_obs=next_obs,
        done=done,
        termination=terminated
    )
    buffer_state = add_transition(buffer_state, t)

    agent_config = agent_state.agent_config
    should_update = jnp.logical_and(
        idx >= agent_config.warmup_steps,
        buffer_state.size >= agent_config.batch_size
    )

    def do_update(agent_state, rng):
        for _ in range(agent_config.update_steps):
            rng, _rng = jax.random.split(rng)
            batch = sample_transitions(buffer_state, agent_config.batch_size, _rng)
            rng, _rng = jax.random.split(rng)
            agent_state, loss_info = agent.update(agent_state, batch, _rng)
        return agent_state, rng, loss_info

    def skip_update(agent_state, rng):
        loss_info = {
            "critic_loss": jnp.array(0.0),
            "actor_loss": jnp.array(0.0),
            "alpha_loss": jnp.array(0.0),
            "alpha": jnp.array(0.0),
        }
        return agent_state, rng, loss_info

    agent_state, rng, loss_info = jax.lax.cond(
        should_update, do_update, skip_update, agent_state, rng
    )

    return agent_state, buffer_state, rng, loss_info


def exp_step(
    runner_state,
    env,
    idx,
    agent: Agent
):
    agent_state, replay_buffer_state, obs, rng = runner_state

    action, rng = agent.step(agent_state, obs[None], rng)
    if len(action.shape) > 1:
        action = action.squeeze(axis=0)
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    agent_state, buffer_state, rng, loss_info = post_env_step(
        agent, agent_state, replay_buffer_state, obs, action, reward, next_obs, done, terminated, rng, idx
    )

    if done:
        next_obs, _ = env.reset()

    agent_config = agent_state.agent_config
    updated = idx >= agent_config.warmup_steps and buffer_state.size >= agent_config.batch_size

    runner_state = (agent_state, buffer_state, next_obs, rng)
    return runner_state, {'episode_info': info, 'loss_info': loss_info, 'env_steps': idx, 'updated': updated}


def experiment(config: ExpConfig, agent: Agent):
    agent_config = config.agent_config
    env_config = config.env_config
    rng = jax.random.PRNGKey(config.exp_seed)

    env = make_env(env_config, agent_config.gamma)
    obs, _ = env.reset()

    action_dim = None
    if env_config.continuous_action:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    
    rng, _rng = jax.random.split(rng)
    agent_state = agent.init_state(
        agent_config,
        action_dim,
        env.observation_space.shape,
        env_config.continuous_action,
        _rng
    )

    replay_buffer_state = init_buffer(
        capacity=agent_config.buffer_capacity,
        obs_shape=env.observation_space.shape,
        action_dim=action_dim
    )

    runner_state = (agent_state, replay_buffer_state, obs, rng)
    log_interval = 10_000
    result = {}
    for i in range(agent_config.total_steps):
        runner_state, result = exp_step(runner_state, env, i, agent)
        if i % log_interval == 0 or 'episode' in result['episode_info']:
            wandb_sac_logging(result)
        if i % log_interval == 0:
            loss_info = result.get('loss_info', {})
            loss_str = (
                f"  critic={loss_info['critic_loss']:.4f}"
                f"  actor={loss_info['actor_loss']:.4f}"
                f"  alpha={loss_info['alpha']:.4f}"
                if result.get('updated') else "  (warmup)"
            )
            print(f"step {i:>8}/{agent_config.total_steps}{loss_str}")

    return {"runner_state": runner_state, "result": result}

def define_metrics():
    wandb.define_metric("env_steps")
    wandb.define_metric("undiscounted_return", step_metric="env_steps")
    wandb.define_metric("episode_length",      step_metric="env_steps")
    wandb.define_metric("critic_loss",         step_metric="env_steps")
    wandb.define_metric("actor_loss",          step_metric="env_steps")
    wandb.define_metric("alpha_loss",          step_metric="env_steps")
    wandb.define_metric("alpha",               step_metric="env_steps")

if __name__ == "__main__":
    main(
        experiment, SACAgent, define_metrics,
        default_config_path="gtd_algos/exp_configs/mujoco_sac.yaml"
    )


