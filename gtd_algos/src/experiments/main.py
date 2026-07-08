import argparse
import time
from typing import Callable

import gymnasium as gym
import numpy as np
import jax
import wandb

from gtd_algos.src.algorithms.agent import Agent
from gtd_algos.src.configs.ExpConfig import ExpConfig
from gtd_algos.src.envs.gym_envs_wrappers import StoreEpisodeReturnsAndLengths


def main(
        experiment: Callable[[ExpConfig, Agent], gym.Env],
        agent: Agent,
        define_metrics: Callable[[None], None],
        default_config_path: str = None,
    ):
    # Reading config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=default_config_path)
    args = parser.parse_args()
    config = ExpConfig.from_yaml(args.config_file)
    default_name = f"{config.algo}_{config.env_config.env_name}_{config.tag}_s{config.exp_seed}"
    run_name = config.run_name or default_name
    print(f"{'='*60}")
    print(f"Run: {run_name}")
    print(f"Env:   {config.env_config.env_name}")
    print(f"Agent: {config.algo}")
    print(f"Agent config: {config.agent_config.d}")
    print(f"Env config:   {config.env_config.d}")
    print(f"{'='*60}")
    ### wandb init
    wandb.init(config=config, project=config.wandb_project_name, name=run_name)
    define_metrics()
    ### start experiment
    start_time = time.time()
    env = jax.block_until_ready(experiment(config, agent))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Time elapsed: {elapsed_time / 60:.2f} minutes')
    total_steps = config.agent_config.total_steps
    wandb.run.summary['SPS'] = int(total_steps / elapsed_time)
    wandb.finish()
