import gymnasium as gym
import numpy as np
import pandas as pd
import numpy.typing as npt

from typing import Any, SupportsFloat

from jsp_vis.rgb_array import gantt_chart_rgb_array

from demonstrator.crf_step_two_input import cp_solver_output, cp_solver_output_mini, worker_availabilities, \
    geometry_line_mapping, human_factor


class CrfWorkerAllocationEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
    }

    # attributes (will be initialized in the constructor)
    # _state: pd.DataFrame
    # _initial_state: pd.DataFrame

    def __init__(self, *,
                 previous_step_output: list[dict[str, Any]],
                 worker_availabilities: list[dict[str, Any]],
                 geometry_line_mapping: list[dict[str, Any]],
                 human_factor_data: dict[str, Any],
                 ):
        state = self._init_state_dataframe(
            previous_step_output=previous_step_output,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
        )

        self._state = None
        self._initial_state = None

    def _init_state_dataframe(self,
                              previous_step_output: list[dict[str, Any]],
                              worker_availabilities: list[dict[str, Any]],
                              geometry_line_mapping: list[dict[str, Any]],
                              human_factor_data: dict[str, Any],
                              ) -> pd.DataFrame:
        pass

    def get_state(self) -> pd.DataFrame:
        return self._state

    def load_state(self, state: pd.DataFrame) -> None:
        # this method does not check if the state is valid or not
        # todo: add a check for the validity of the state

        self._state = state

    def step(self, action: int) -> (npt.NDArray, SupportsFloat, bool, bool, dict[str, Any]):
        pass

    def reset(self, **kwargs) -> (npt.NDArray, dict[str, Any]):
        super().reset(**kwargs)

        return np.array(self._state), {}

    def render(self, mode='human', **kwargs) -> Any:
        sate = self.get_state()

    def is_terminal_state(self) -> bool:
        return False

    def valid_action_mask(self) -> npt.NDArray[np.int8]:
        # NOTE: 0 intentionally included
        raise NotImplementedError()
        return np.array([row[self._valid_action_column_idx] for row in self._state[:-1]], dtype=np.int8)

    def valid_action_list(self) -> list[int]:
        return [i for i, is_valid in enumerate(self.valid_action_mask()) if is_valid]

    def random_rollout(self) -> int:
        done = self.is_terminal_state()

        # unfortunately, we dont have any information about the past rewards
        # so we just return the cumulative reward from the current state onwards
        cumulative_reward_from_current_state_onwards = 0

        while not done:
            valid_action_list = self.valid_action_list()
            random_action = np.random.choice(valid_action_list)
            _, rew, done, _, _ = self.step(random_action)
            cumulative_reward_from_current_state_onwards += rew

        return cumulative_reward_from_current_state_onwards


if __name__ == '__main__':
    step_1_output = cp_solver_output_mini
    worker_availabilities = worker_availabilities
    geometry_line_mapping = geometry_line_mapping
    human_factor_data = human_factor

    env = CrfWorkerAllocationEnv(
        previous_step_output=step_1_output,
        worker_availabilities=worker_availabilities,
        geometry_line_mapping=geometry_line_mapping,
        human_factor_data=human_factor_data,
    )
