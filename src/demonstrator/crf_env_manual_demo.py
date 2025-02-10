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
    log.info("each task/node corresponds to an action")

    while not done:
        env.render()
        allocations_dict = env.get_worker_allocation(filter_no_workers_assigned=True)
        log.info(pprint.pformat(allocations_dict))
        current_interval_from = None
        current_interval_to = None

        # get the current interval by getting a row where the col 'is_current_interval' is 1
        for row in range(env.get_state().shape[0]):
            if env.get_state().at[row, 'is_current_interval'] == 1:
                current_interval_from = env.get_state().at[row, 'interval_start']
                current_interval_to = env.get_state().at[row, 'interval_end']
                break

        questions = [
            inquirer.List(
                "next_action",
                message=f"Which Worker should be scheduled next? The current interval is from {current_interval_from} to {current_interval_to}",
                choices=[
                    (f"Allocate Worker '{env._idx_to_worker_map[worker].replace('worker_', '')}' ({env.get_state().at[row, env._idx_to_worker_map[worker]]}) on line '{env.get_state().at[row, 'line']}' (Row with idx={row})", (row, worker))
                    for (row, worker) in env.valid_action_tuples()
                ],
            ),
        ]
        next_action_tuple = inquirer.prompt(questions)["next_action"]
        next_action = env.action_tuple_to_action_idx(next_action_tuple)

        msg = (f"Allocating worker '{env._idx_to_worker_map[next_action_tuple[1]].replace('worker_', '')}' "
               f"on line '{env.get_state().at[next_action_tuple[0], 'line']}' for task {env.get_state().at[next_action_tuple[0], 'Task']}")
        log.info(msg)

        _, reward, done, _, _ = env.step(next_action)



