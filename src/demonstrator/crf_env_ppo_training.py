import stable_baselines3 as sb3
import sb3_contrib
import datetime

import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.logger import log

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

from demonstrator.crf_worker_allocation_gym_env import CrfWorkerAllocationEnv

from demonstrator.crf_step_two_input import (cp_solver_output, cp_solver_output2, worker_availabilities,
                                             geometry_line_mapping, human_factor)

# Define the environment creation function
worker_availabilities = worker_availabilities
geometry_line_mapping = geometry_line_mapping
human_factor_data = human_factor

start_timestamp = 1693807200
step_1_output = cp_solver_output2
def make_env():


    env = CrfWorkerAllocationEnv(
        previous_step_output=step_1_output['allocations'],
        worker_availabilities=worker_availabilities,
        geometry_line_mapping=geometry_line_mapping,
        human_factor_data=human_factor_data,
        start_timestamp=start_timestamp,
        allocate_workers_on_the_same_line_if_possible=False,
    )

    def mask_fn(env: gym.Env) -> np.ndarray:
        return env.unwrapped.valid_action_mask()

    env = ActionMasker(env, mask_fn)

    env = Monitor(env)
    return env

# Create the vectorized environment
vec_env = make_vec_env(make_env, n_envs=4, vec_env_cls=DummyVecEnv)


model = sb3_contrib.MaskablePPO(MaskableActorCriticPolicy, vec_env, verbose=1, device="auto")

if __name__ == '__main__':
    # Train the agent
    log.info("training the model")
    model.learn(total_timesteps=1_000)
    model.save(
        f"crf_rl_model-action-{vec_env.action_space.shape}_obs-{vec_env.observation.shape}_date-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
