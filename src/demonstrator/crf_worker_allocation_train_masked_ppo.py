import gymnasium as gym
import sb3_contrib
import numpy as np
import stable_baselines3 as sb3

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from demonstrator.crf_step_two_input import cp_solver_output2
from demonstrator.crf_worker_allocation_gym_env import CrfWorkerAllocationEnv

from demonstrator.crf_step_two_input import (cp_solver_output, cp_solver_output2, worker_availabilities,
                                             geometry_line_mapping, human_factor)
from utils.logger import log

start_timestamp = 1693807200
step_1_output = cp_solver_output2

worker_availabilities = worker_availabilities
geometry_line_mapping = geometry_line_mapping
human_factor_data = human_factor


env = CrfWorkerAllocationEnv(
        previous_step_output=step_1_output,
        worker_availabilities=worker_availabilities,
        geometry_line_mapping=geometry_line_mapping,
        human_factor_data=human_factor_data,
        start_timestamp=start_timestamp,
        allocate_workers_on_the_same_line_if_possible=False,
)
env = sb3.common.monitor.Monitor(env)


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.valid_action_mask()


env = ActionMasker(env, mask_fn)

model = sb3_contrib.MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)

if __name__ == '__main__':
    # Train the agent
    log.info("training the model")
    model.learn(total_timesteps=10_000)