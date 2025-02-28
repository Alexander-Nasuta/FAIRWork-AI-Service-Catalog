import pprint

import inquirer
import sb3_contrib

from demonstrator.crf_worker_allocation_gym_env import CrfWorkerAllocationEnv
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from utils.logger import log

from demonstrator.crf_step_two_input import (cp_solver_output, cp_solver_output2, worker_availabilities,
                                             geometry_line_mapping, human_factor)

worker_availabilities = worker_availabilities
geometry_line_mapping = geometry_line_mapping
human_factor_data = human_factor

start_timestamp = 1693807200
step_1_output = cp_solver_output2

env = CrfWorkerAllocationEnv(
    previous_step_output=step_1_output['allocations'],
    worker_availabilities=worker_availabilities,
    geometry_line_mapping=geometry_line_mapping,
    human_factor_data=human_factor_data,
    start_timestamp=start_timestamp,
    allocate_workers_on_the_same_line_if_possible=False,
)
env.reset()

if __name__ == '__main__':

    done = False
    log.info("each task/node corresponds to an action")

    # load model from file
    model = sb3_contrib.MaskablePPO.load("crf_rl_model-action-()_obs-(110, 241).zip")

    obs, _ = env.reset()
    while not done:
        masks = env.valid_action_mask()
        action, _ = model.predict(observation=obs, deterministic=True, action_masks=masks)
        obs, rew, done, turn, info = env.step(action)
        log.info(f"Action: {action}, Reward: {rew}")

    res = env.get_worker_allocation(filter_no_workers_assigned=True)
    log.info(pprint.pformat(res))

    experience, resilience, preference = env.get_KPIs()
    log.info(f"Experience: {experience}, Resilience: {resilience}, Preference: {preference}")
