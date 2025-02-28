import pprint

import random
import gymnasium as gym
import pandas as pd
from gymcts.gymcts_agent import SoloMCTSAgent

from gymcts.gymcts_gym_env import SoloMCTSGymEnv

from demonstrator.crf_worker_allocation_gym_env import CrfWorkerAllocationEnv
from utils.logger import log

from demonstrator.crf_step_two_input import (cp_solver_output, cp_solver_output2, worker_availabilities,
                                             geometry_line_mapping, human_factor)


class CrfMCTSWrapper(SoloMCTSGymEnv, gym.Wrapper):

    def __init__(self, env: CrfWorkerAllocationEnv):
        gym.Wrapper.__init__(self, env)

    def load_state(self, state: pd.DataFrame) -> None:
        self.env.unwrapped.load_state(state)

    def is_terminal(self) -> bool:
        return self.env.unwrapped.is_terminal_state()

    def get_valid_actions(self) -> list[int]:
        return self.env.unwrapped.valid_action_list()

    def rollout(self) -> float:
        return self.env.unwrapped.greedy_rollout_sparse()

    def get_state(self) -> pd.DataFrame:
        return env.unwrapped.get_state()


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

env = CrfMCTSWrapper(env)

if __name__ == '__main__':
    done = False
    agent = SoloMCTSAgent(
        env=env,
        clear_mcts_tree_after_step=True,
        render_tree_after_step=True,
        exclude_unvisited_nodes_from_render=True,
        number_of_simulations_per_step=2,
    )
    actions = agent.solve(render_tree_after_step=True)

    env.render()
    experience, resilience, preference = env.get_KPIs()
    log.info(f"KPIs: experience={experience:.2f}, resilience={resilience:.2f}, preference={preference:.2f}")
