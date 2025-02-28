import pprint

import inquirer

from demonstrator.crf_worker_allocation_gym_env import CrfWorkerAllocationEnv
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
    env.greedy_rollout_sparse()
    experience, resilience, preference = env.get_KPIs()
    log.info(f"KPIs: experience={experience:.2f}, resilience={resilience:.2f}, preference={preference:.2f}")
