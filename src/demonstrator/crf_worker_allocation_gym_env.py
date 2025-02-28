import copy
import pprint
from collections import namedtuple
from datetime import datetime
import pytz

import gymnasium as gym
import numpy as np
import pandas as pd
import numpy.typing as npt

from utils.crf_timestamp_solver_time_conversion import timestamp_to_solver_time, solver_time_to_timestamp
from utils.logger import log
from typing import Any, SupportsFloat, List, Tuple, Hashable

from demonstrator.crf_step_two_input import (cp_solver_output, cp_solver_output2, worker_availabilities,
                                             geometry_line_mapping, human_factor)

worker_decision_variables = namedtuple(
    'WorkerVars',
    ['available', 'medical_condition', 'preference', 'resilience', 'experience', 'allocated']
)


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
                 start_timestamp: int,
                 dense_reward: bool = True,
                 preference_weight: float = 1.0,
                 resilience_weight: float = 1.0,
                 experience_weight: float = 1.0,
                 allocate_workers_on_the_same_line_if_possible: bool = True,
                 ):

        df_state = self._init_state_dataframe(
            previous_step_output=previous_step_output,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
        )

        # will be set by load_state
        self._state: pd.DataFrame = None
        self._worker_to_idx_map: dict = None
        self._idx_to_worker_map: dict = None
        self._n_rows: int = None
        self._n_workers: int = None

        # reward settings
        self._dense_reward = dense_reward
        self._preference_weight = preference_weight
        self._resilience_weight = resilience_weight
        self._experience_weight = experience_weight

        self._start_timestamp = start_timestamp

        self._allocate_workers_on_the_same_line_if_possible = allocate_workers_on_the_same_line_if_possible

        self._initial_state = df_state.copy(deep=True)

        self.load_state(state=self._initial_state)

        # the action space is basically a matrix of workers and the rows of the df, but flattened
        # action = 0 is the first worker for the first row
        # action = 1 is the second worker for the first row
        # action = _n_workers + 0 is the first worker for the second row
        # action = _n_workers + 1 is the second worker for the second row
        # ...
        # action = _n_workers * _n_rows - 1 is the last worker for the last row
        possible_actions = self._n_workers * self._n_rows
        self.action_space = gym.spaces.Discrete(possible_actions)

        initial_observation = self._state_as_numpy_array()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=initial_observation.shape,
            dtype=initial_observation.dtype
        )

    def action_tuple_to_action_idx(self, action_tuple: tuple[int, int]) -> int:
        row, worker = action_tuple
        return row * self._n_workers + worker

    def action_idx_to_action_tuple(self, action_idx: int) -> tuple[int, int]:
        return divmod(action_idx, self._n_workers)

    @staticmethod
    def _init_state_dataframe(previous_step_output: list[dict[str, Any]],
                              worker_availabilities: list[dict[str, Any]],
                              geometry_line_mapping: list[dict[str, Any]],
                              human_factor_data: dict[str, Any],
                              start_timestamp: int,
                              ) -> pd.DataFrame:

        line_allocations = previous_step_output

        # map line allocation to solver time domain
        line_allocations = [
            elem | {
                'Start_solver_time': timestamp_to_solver_time(elem['Start'], start_timestamp),
                'Finish_solver_time': timestamp_to_solver_time(elem['Finish'], start_timestamp),
            } for elem in line_allocations
        ]

        # map worker availabilities to solver time domain
        worker_availabilities = [
            elem | {
                'from_solver_time': timestamp_to_solver_time(elem['from_timestamp'], start_timestamp),
                'end_solver_time': timestamp_to_solver_time(elem['end_timestamp'], start_timestamp),
            } for elem in worker_availabilities
        ]

        relevant_intervals = CrfWorkerAllocationEnv._get_relevant_intervals(
            line_allocations=line_allocations,
            worker_availabilities=worker_availabilities,
            start_timestamp=start_timestamp,
        )
        log.info(f"relevant_intervals: {relevant_intervals}")

        df_data = []
        for interval_idx, (interval_start, interval_end) in enumerate(relevant_intervals):
            line_allocations_within_interval = [
                elem for elem in line_allocations
                if
                CrfWorkerAllocationEnv._intervals_overlap(
                    (elem['Start_solver_time'], elem['Finish_solver_time']),
                    (interval_start, interval_end)
                )
            ]
            workers_within_interval = [
                elem for elem in worker_availabilities
                if
                CrfWorkerAllocationEnv._intervals_overlap(
                    (elem['from_solver_time'], elem['end_solver_time']),
                    (interval_start, interval_end)
                )
            ]

            for line_elem in line_allocations_within_interval:

                required_workers = CrfWorkerAllocationEnv._get_required_number_of_workers(
                    line=line_elem['Resource'],
                    geometry=line_elem['geometry'],
                    geometry_line_mapping=geometry_line_mapping
                )
                log.debug(f"required workers for line '{line_elem['Resource']}' and geometry '{line_elem['geometry']}' "
                          f"is {required_workers}.")

                data_row_dict = {
                    'interval_no': interval_idx,
                    'is_current_interval': 1 if interval_idx == 0 else 0,
                    'interval_start': interval_start,
                    'interval_end': interval_end,
                    'Task': line_elem['Task'],
                    'Task_interval': f"{line_elem['Task']} × Interval {interval_idx}",
                    # todo: check if timstamp is needed instead of interval idx
                    'line': line_elem['Resource'],
                    'geometry': line_elem['geometry'],
                    'required_workers': required_workers,
                    'allocated_workers': 0,
                    'required_workers_met': 0,
                    'row_done': 0,
                }
                n_workers_available_for_this_task = 0
                for worker_in_interval in workers_within_interval:
                    worker_id = worker_in_interval['worker']
                    preference, resilience, medical_condition, experience = CrfWorkerAllocationEnv._get_human_factor_data(
                        worker=worker_id,
                        geometry=line_elem['geometry'],
                        human_factor_data=human_factor_data
                    )
                    n_workers_available_for_this_task += 1
                    res = worker_decision_variables(
                        available=1,
                        medical_condition=int(medical_condition),
                        preference=preference,
                        resilience=resilience,
                        experience=experience,
                        allocated=0
                    )
                    data_row_dict = data_row_dict | {
                        f'worker_{worker_id}': res
                    }
                df_data.append(data_row_dict)

        df = pd.DataFrame(df_data)
        return df

    @staticmethod
    def _get_human_factor_data(
            worker: str,
            geometry: str,
            human_factor_data: dict[str, Any]) -> (float, float, bool, float):
        for elem in human_factor_data:
            if elem['geometry'] == geometry and elem['worker'] == worker:
                preference = elem['preference']
                resilience = elem['resilience']
                medical_condition = elem['medical_condition'].lower() == "true"
                experience = elem['experience']

                log.debug(
                    f"human factor data found for worker '{worker}' and geometry '{geometry}': {(preference, resilience, medical_condition, experience)}")

                return preference, resilience, medical_condition, experience
        else:
            log.warning(f"no human factor data found for worker '{worker}' and geometry '{geometry}'. "
                        f"Returning default values of (0.5, 0.5, True, 0.5)")
            return 0.5, 0.5, True, 0.5

    @staticmethod
    def _human_readable_timestamp(timestamp: int | float) -> str:
        human_readable_time = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
        return human_readable_time.strftime('%A %Y-%m-%d %H:%M:%S')

    @staticmethod
    def _get_required_number_of_workers(line: str, geometry: str, geometry_line_mapping: list[dict[str, Any]]) -> int:
        for elem in geometry_line_mapping:
            if elem['geometry'] == geometry:
                if elem['main_line'] == line or line in elem['alternative_lines']:
                    return elem['number_of_workers']
        else:
            log.warning(f"no line allocation found for line {line} and geo {geometry}. Returning a default value of 4")
            return 4

    @staticmethod
    def _intervals_overlap(interval1, interval2):
        start1, end1 = interval1
        start2, end2 = interval2
        return start1 < end2 and start2 < end1

    @staticmethod
    def _get_relevant_intervals(
            line_allocations: list[dict],
            worker_availabilities: list[dict],
            start_timestamp: int) -> list[tuple[int, int]]:

        interval_bounds = set()

        for line_allocation_elem in line_allocations:
            interval_bounds.add(line_allocation_elem['Start_solver_time'])
            interval_bounds.add(line_allocation_elem['Finish_solver_time'])

        for worker_availabilities_elem in worker_availabilities:
            interval_bounds.add(worker_availabilities_elem['from_solver_time'])
            interval_bounds.add(worker_availabilities_elem['end_solver_time'])

        interval_bounds_ascending_list = sorted(list(interval_bounds))

        interval_tuple = []  # [(start, end), ...]
        for interval_start, interval_end in zip(interval_bounds_ascending_list[:-1],
                                                interval_bounds_ascending_list[1:]):
            interval_tuple.append((interval_start, interval_end))

        n_intervals = len(interval_tuple)
        log.debug(f"the schedule is devided into {n_intervals} intervals: {interval_tuple}")

        relevant_intervals = []

        for interval_idx, (interval_start, interval_end) in enumerate(interval_tuple):
            log.debug(f"interval {interval_idx}: {interval_start} - {interval_end} "
                      f"({solver_time_to_timestamp(interval_start, start_timestamp)}-{solver_time_to_timestamp(interval_end, start_timestamp)})")
            log.debug(f"interval {interval_idx}: "
                      f"{CrfWorkerAllocationEnv._human_readable_timestamp(solver_time_to_timestamp(interval_start, start_timestamp))} "
                      f"- {CrfWorkerAllocationEnv._human_readable_timestamp(solver_time_to_timestamp(interval_end, start_timestamp))}")
            # find line allocations that are within this interval
            line_allocations_within_interval = [
                elem for elem in line_allocations
                if
                CrfWorkerAllocationEnv._intervals_overlap((elem['Start_solver_time'], elem['Finish_solver_time']),
                                                          (interval_start, interval_end))
            ]
            log.debug("line allocations within this interval:")
            log.debug(pprint.pformat(line_allocations_within_interval))
            log.debug("available workers within this interval:")

            workers_within_interval = [
                elem for elem in worker_availabilities
                if
                CrfWorkerAllocationEnv._intervals_overlap(
                    (elem['from_solver_time'], elem['end_solver_time']),
                    (interval_start, interval_end)
                )
            ]
            log.debug(pprint.pformat(workers_within_interval))

            if len(workers_within_interval) and len(line_allocations_within_interval):
                relevant_intervals.append((interval_start, interval_end))

        return relevant_intervals

    def get_state(self) -> pd.DataFrame:
        return self._state

    def load_state(self, state: pd.DataFrame) -> None:
        # this method does not check if the state is valid or not
        # todo: add a check for the validity of the state
        self._state = state.copy(deep=True)

        worker_to_idx_map = {col: idx for idx, col in enumerate(sorted(self._get_all_worker_cols()))}
        idx_to_worker_map = {idx: col for col, idx in worker_to_idx_map.items()}

        self._worker_to_idx_map = worker_to_idx_map
        self._idx_to_worker_map = idx_to_worker_map

        self._n_rows = self._state.shape[0]
        self._n_workers = len(worker_to_idx_map)


    def _calculate_reward(
            self,
            named_worker_tuple: tuple,
            is_terminal: bool,
            row_done_without_meeting_required_workers_penalty=0
    ) -> SupportsFloat:
        if self._dense_reward:
            reward = 0
            reward += self._preference_weight * named_worker_tuple.preference
            reward += self._resilience_weight * named_worker_tuple.resilience
            reward += self._experience_weight * named_worker_tuple.experience
            reward += row_done_without_meeting_required_workers_penalty
            return reward
        else:
            if is_terminal:
                experience, resilience, preference = env.get_KPIs()
                weighted_sum = self._preference_weight * preference + self._resilience_weight * resilience + self._experience_weight * experience
                scaled_weighted_sum = weighted_sum / (
                        self._preference_weight + self._resilience_weight + self._experience_weight)
                return scaled_weighted_sum
            else:
                return 0

    def step(self, action: int) -> (npt.NDArray, SupportsFloat, bool, bool, dict[str, Any]):
        # convert action to tuple
        action_row, action_worker = self.action_idx_to_action_tuple(action)
        # log.info(f"performing action: {action} ({action_row}, {action_worker})")

        # check if the action is valid
        # 1. 'is_current_interval' has to be 1 in the specified row
        # 2.  the worker has to be available and not allocated and has to have a medical condition of True
        if self._state.at[action_row, 'is_current_interval'] == 0:
            # log.warning(f"the interval is not current, so the action is invalid.")
            return self._state_as_numpy_array(), 0, self.is_terminal_state(), False, {}

        # if the row is done, then the action is invalid
        if self._state.at[action_row, 'row_done'] == 1:
            # log.warning(f"the row is done, so the action is invalid.")
            return self._state_as_numpy_array(), 0, self.is_terminal_state(), False, {}

        worker_col = self._idx_to_worker_map[action_worker]

        # worker_decision_variables = namedtuple(
        #     'WorkerVars',
        #     ['available', 'medical_condition', 'preference', 'resilience', 'experience', 'allocated']
        # )
        worker_named_tuple = self._state.at[action_row, worker_col]

        if not isinstance(worker_named_tuple, tuple):
            # log.warning(f"the worker is not available, so the action is invalid.")
            return self._state_as_numpy_array(), 0, self.is_terminal_state(), False, {}

        if not worker_named_tuple.available:
            # log.warning(f"the worker is not available, so the action is invalid.")
            return self._state_as_numpy_array(), 0, self.is_terminal_state(), False, {}

        if worker_named_tuple.allocated:
            # log.warning(f"the worker is already allocated, so the action is invalid.")
            return self._state_as_numpy_array(), 0, self.is_terminal_state(), False, {}

        if not worker_named_tuple.medical_condition:
            # log.warning(f"the worker has a medical condition of False, so the action is invalid.")
            return self._state_as_numpy_array(), 0, self.is_terminal_state(), False, {}

        # at this point, the action is valid
        # so we can update the state

        # 1.1 set the worker as allocated in the specified row
        # 1.2 update the allocated_workers and required_workers_met in the specified row
        #    -> required_workers_met is 1 if the allocated_workers == required_workers
        #    -> allocated_workers is incremented by 1
        # 1.3 set the worker as not available in all rows with is_current_interval == 1
        # 1.4 check if the action_row is done
        #    -> if required_workers_met == 1, then set row_done to 1
        #    -> if there are no more workers available for this row, then set row_done to 1

        # 1.1
        named_tuple = self._state.at[action_row, worker_col]
        allocated_updated_named_tuple = named_tuple._replace(allocated=1)
        self._state.at[action_row, worker_col] = allocated_updated_named_tuple
        # 1.2
        self._state.at[action_row, 'allocated_workers'] += 1
        row_done_without_meeting_required_workers_penalty = 0
        if self._state.at[action_row, 'allocated_workers'] == self._state.at[action_row, 'required_workers']:
            self._state.at[action_row, 'required_workers_met'] = 1
            # 1.4
            self._state.at[action_row, 'row_done'] = 1
        else:
            valid_actions_for_row = self.valid_action_tuples_for_row(action_row)
            if not len(valid_actions_for_row):
                self._state.at[action_row, 'row_done'] = 1
                row_done_without_meeting_required_workers_penalty = -10
            # check also for the remaining rows with is_current_interval == 1 and row_done == 0 if there are no more
            # workers available
            for row_idx, row in self._state.iterrows():
                if row['is_current_interval'] == 1:
                    valid_actions_for_row = self.valid_action_tuples_for_row(row_idx)
                    if not len(valid_actions_for_row):
                        self._state.at[row_idx, 'row_done'] = 1
                        row_done_without_meeting_required_workers_penalty = -10

        # 1.3
        for row_idx, action_row in self._state.iterrows():
            if action_row['is_current_interval'] == 1:
                named_tuple = self._state.at[row_idx, worker_col]
                updated_named_tuple = named_tuple._replace(available=0)
                self._state.at[row_idx, worker_col] = updated_named_tuple

        # check if we need to go to the next interval
        # we go to the next interval if all rows where is_current_interval == 1 are done, i.e. row_done == 1
        goto_next_interval = self._state[self._state['is_current_interval'] == 1]['row_done'].all()
        # log.info(f"goto_next_interval: {goto_next_interval}")

        nested_reward = 0

        # todo: find a more elegant way for going to the next interval in the terminal corner case
        all_rows_done = True
        for row_idx, row in self.get_state().iterrows():
            if row['is_current_interval'] == 1:
                valid_actions = self.valid_action_tuples_for_row(row_idx)
                # log.info(f"valid actions for row {row_idx}: {valid_actions}")
                if len(valid_actions):
                    all_rows_done = False
                    break
        if all_rows_done:
            goto_next_interval = True

        # if valid action mask has only zeros, then goto_next_interval
        if not self.valid_action_mask().any():
            goto_next_interval = True

        if goto_next_interval:
            prev_interval_no = self._state[self._state['is_current_interval'] == 1]['interval_no'].iloc[0]
            next_interval_no = prev_interval_no + 1

            # log.info(f"transitioning form interval {prev_interval_no} to interval {next_interval_no}.")

            # set all rows with interval_no == current_interval_no to is_current_interval = 0
            self._state.loc[self._state['interval_no'] == prev_interval_no, 'is_current_interval'] = 0

            # set all rows with interval_no == next_interval_no to is_current_interval = 1
            self._state.loc[self._state['interval_no'] == next_interval_no, 'is_current_interval'] = 1

            # check if we are in the last interval
            if self.is_terminal_state():
                # print("terminal state reached")
                reward = self._calculate_reward(
                    named_worker_tuple=allocated_updated_named_tuple,
                    is_terminal=True,
                    row_done_without_meeting_required_workers_penalty=row_done_without_meeting_required_workers_penalty
                )
                return self._state_as_numpy_array(), nested_reward + reward, True, False, {}

            if self._allocate_workers_on_the_same_line_if_possible:
                # if
                # - a worker was allocated (named_tuple.allocated == 1) in a row with interval_no == prev_interval_no to a Task specific T
                # - and the worker is available a row with is_current_interval == 1 for the same Task T
                # then set the worker as not available in all rows with is_current_interval == 1
                # and set the worker as allocated in the row with is_current_interval == 1 and Task == T

                for prev_row_idx, prev_row in self._state.iterrows():
                    if prev_row['interval_no'] != prev_interval_no:
                        continue
                    # row where interval_no == prev_interval_no

                    for next_row_idx, next_row in self._state.iterrows():
                        if next_row['interval_no'] != next_interval_no:
                            continue
                        # row where interval_no == next_interval_no

                        if next_row['interval_no'] != next_interval_no:
                            continue

                        if prev_row['Task'] != next_row['Task']:
                            continue

                        task_candidate = next_row_idx
                        worker_candidate = self._get_allocated_workers_in_row(prev_row_idx)

                        # create tuples with task_candidate as the first element and worker_candidate entries
                        # as the second element
                        candidates = [(task_candidate, worker) for worker in worker_candidate]

                        # perform a step with all candidates
                        # not valid states will be ignored by the step function anyway, so there is no need to check
                        for candidate_tuple in candidates:
                            candidate_action = self.action_tuple_to_action_idx(candidate_tuple)
                            _, rew, _, _, _ = self.step(candidate_action)
                            nested_reward += rew

        is_terminal = self.is_terminal_state()
        reward = self._calculate_reward(
            named_worker_tuple=allocated_updated_named_tuple,
            is_terminal=is_terminal,
            row_done_without_meeting_required_workers_penalty=row_done_without_meeting_required_workers_penalty
        )
        return self._state_as_numpy_array(), nested_reward + reward, is_terminal, False, {}

    def reset(self, **kwargs) -> (npt.NDArray, dict[str, Any]):
        super().reset(**kwargs)
        self.load_state(state=self._initial_state)
        return self._state_as_numpy_array(), {}

    def _get_all_worker_cols(self) -> list[str]:
        worker_ids = []
        for col in self._state.columns:
            if col.startswith('worker_'):
                worker_ids.append(col)
        return worker_ids

    def _get_number_of_workers(self) -> int:
        return len(self._get_all_worker_cols())

    def _state_as_numpy_array(self) -> npt.NDArray:
        df = self.get_state().copy()

        # remove columns that are not needed
        df = df.drop(columns=[
            'Task',
            'Task_interval',
            'line',
            'geometry',
            'interval_no',
        ])

        # tuple_columns = [col for col in df.columns if df[col].dtype == 'object']
        tuple_columns = self._get_all_worker_cols()

        # fill NaN values with (1, 1, 0.22, 0.93, 0.56, 0)
        for col in tuple_columns:
            df[col] = df[col].fillna(
                pd.Series([
                    (0, 0, 0, 0, 0, 0)
                    for _ in range(df.shape[0])
                ])
            )

        # Expand each tuple column into separate columns
        for col in tuple_columns:
            expanded_df = df[col].apply(pd.Series)
            expanded_df.columns = [f'{col}_{i}' for i in range(expanded_df.shape[1])]
            df = pd.concat([df.drop(columns=[col]), expanded_df], axis=1)

        # Convert the final DataFrame to a NumPy array
        return df.to_numpy(dtype=np.float32)

    def render(self, mode='human', **kwargs) -> Any:
        df = self.get_state().copy()
        # add index of the row as a column
        df['idx'] = df.index
        df = df[df['is_current_interval'] == 1]
        renaming_dict = {
            'interval_start': 'i_start',
            'interval_end': 'i_end',
            'required_workers': 'req_w',
            'allocated_workers': 'alloc_w',
            'required_workers_met': 'req_w_met',
            'interval_no': 'no.',
        }
        cols_to_display = [
            'idx',
            'no.',
            'i_start',
            'i_end',
            'line',
            # 'geometry',
            'req_w',
            'alloc_w',
            'req_w_met',
        ]
        # renaming cols, so more fit into the console
        df = df.rename(columns=renaming_dict)
        # add worker columns that have not NaN values to cols_to_display
        n_visualized = 0
        for col in df.columns:
            if col.startswith('worker_') and not df[col].isna().all():
                cols_to_display.append(col)
                n_visualized += 1
            if n_visualized >= 4:
                break
        print(self._worker_to_idx_map)
        print(df[cols_to_display].to_string())

    def is_terminal_state(self) -> bool:
        # the terminal state is reached when is_current_interval is 0 for all rows
        # return not self._state['is_current_interval'].any()
        return not len(self.valid_action_tuples())

    def _get_allocated_workers_in_row(self, row_idx: int) -> list[str]:
        # find all columns that start with 'worker_' and have allocated == 1
        # map them to the worker id
        allocated_workers = []
        row = self._state.loc[row_idx]
        for worker, worker_idx in self._worker_to_idx_map.items():
            if not isinstance(row[worker], tuple):
                continue
            if row[worker].allocated == 1:
                allocated_workers.append(worker_idx)

        return allocated_workers

    def valid_action_tuples_for_row(self, row_idx: int) -> list[tuple[int, int]]:
        # find all valid actions as tuple (row, worker)
        valid_actions = []
        # a valid action is a tuple (row, worker) where
        # is_current_interval is 1
        # the worker is available
        # the worker is not allocated
        # the worker medical condition is True (or 1)
        row = self._state.loc[row_idx]
        if row['is_current_interval'] == 1:
            for worker, worker_idx in self._worker_to_idx_map.items():
                # if row[worker] is NaN, then continue
                if not isinstance(row[worker], tuple):
                    continue
                if row[worker].allocated == 0 and row[worker].available == 1 and row[worker].medical_condition:
                    valid_actions.append((int(row_idx), int(worker_idx)))

        return valid_actions

    def valid_action_tuples(self) -> list[tuple[int, int]]:
        # find all valid actions as tuple (row, worker)
        valid_actions = []
        # a valid action is a tuple (row, worker) where
        # is_current_interval is 1
        # the worker is available
        # the worker is not allocated
        # the worker medical condition is True (or 1)
        for row_idx, row in self._state.iterrows():
            if row['row_done'] == 1:
                continue
            if row['is_current_interval'] == 1:
                for worker, worker_idx in self._worker_to_idx_map.items():
                    # if row[worker] is NaN, then continue
                    if not isinstance(row[worker], tuple):
                        continue
                    if row[worker].allocated == 0 and row[worker].available == 1 and row[worker].medical_condition:
                        valid_actions.append((int(row_idx), int(worker_idx)))

        return valid_actions

    def valid_action_mask(self) -> npt.NDArray[np.int8]:
        valid_action_tuples = self.valid_action_tuples()
        valid_actions = [self.action_tuple_to_action_idx(action_tuple) for action_tuple in valid_action_tuples]
        valid_action_mask = [1 if i in valid_actions else 0 for i in range(self.action_space.n)]
        return np.array(valid_action_mask, dtype=np.int8)

    def valid_action_list(self) -> list[int]:
        return [i for i, is_valid in enumerate(self.valid_action_mask()) if is_valid]

    def random_rollout(self) -> int:
        done = len(self.valid_action_mask())

        # unfortunately, we dont have any information about the past rewards
        # so we just return the cumulative reward from the current state onwards
        cumulative_reward_from_current_state_onwards = 0

        while not done:
            valid_action_list = self.valid_action_list()
            random_action = np.random.choice(valid_action_list)
            _, rew, done, _, _ = self.step(random_action)
            cumulative_reward_from_current_state_onwards += rew

        return cumulative_reward_from_current_state_onwards

    def get_number_of_workers(self) -> int:
        return self._n_workers

    def get_number_of_intervals(self) -> int:
        return self._state['interval_no'].nunique()

    def _get_KPI_highscore(self):
        possible_high_score = 0
        total_time = self._state[self._state['interval_no'] == self.get_number_of_intervals() - 1]['interval_end'].max()
        for row_idx, row in self._state.iterrows():
            interval_length = row['interval_end'] - row['interval_start']
            weigth = interval_length / total_time

            required_workers = row['required_workers']

            possible_high_score += required_workers * weigth
        return possible_high_score

    def get_KPIs(self) -> (float, float, float):

        score_experience = 0
        score_resilience = 0
        score_preference = 0

        possible_high_score = 0

        # total_time is the maximum time of the last interval
        total_time = self._state[self._state['interval_no'] == self.get_number_of_intervals() - 1]['interval_end'].max()

        for row_idx, row in self._state.iterrows():
            interval_length = row['interval_end'] - row['interval_start']
            weigth = interval_length / total_time

            required_workers = row['required_workers']

            possible_high_score += required_workers * weigth

            row_score_experience = 0
            row_score_resilience = 0
            row_score_preference = 0

            for worker, worker_idx in self._worker_to_idx_map.items():
                if not isinstance(row[worker], tuple):
                    continue
                if row[worker].allocated == 1:
                    row_score_experience += row[worker].experience
                    row_score_resilience += row[worker].resilience
                    row_score_preference += row[worker].preference

            score_experience += row_score_experience * weigth
            score_resilience += row_score_resilience * weigth
            score_preference += row_score_preference * weigth

        experience = score_experience / possible_high_score
        resilience = score_resilience / possible_high_score
        preference = score_preference / possible_high_score

        return experience, resilience, preference

    def get_scaled_KPI_score(self) -> float:
        experience, resilience, preference = self.get_KPIs()
        weighted_sum = self._preference_weight * preference + self._resilience_weight * resilience + self._experience_weight * experience
        scaled_weighted_sum = weighted_sum / (
                self._preference_weight + self._resilience_weight + self._experience_weight)
        return scaled_weighted_sum

    def best_eager_action(self) -> int | None:
        best_action = None
        best_reward = -np.inf

        for (row, worker) in self.valid_action_tuples():
            action = self.action_tuple_to_action_idx((row, worker))

            worker_tuple = self._state.at[row, self._idx_to_worker_map[worker]]
            worker_preference = worker_tuple.preference
            worker_resilience = worker_tuple.resilience
            worker_experience = worker_tuple.experience

            score = self._preference_weight * worker_preference + self._resilience_weight * worker_resilience + self._experience_weight * worker_experience

            reward_prognoses = score

            if reward_prognoses > best_reward:
                best_reward = reward_prognoses
                best_action = action

        return best_action

    def greedy_rollout_sparse(self) -> int:
        done = not len(self.valid_action_mask())

        # unfortunately, we dont have any information about the past rewards
        # so we just return the cumulative reward from the current state onwards
        cumulative_reward_from_current_state_onwards = 0

        while not done:
            best_action = self.best_eager_action()
            _, rew, done, _, _ = self.step(best_action)
            cumulative_reward_from_current_state_onwards += rew
            # print(f"best action: {best_action}, reward: {rew}")

        experience, resilience, preference = self.get_KPIs()
        weighted_sum = self._preference_weight * preference + self._resilience_weight * resilience + self._experience_weight * experience
        scaled_weighted_sum = weighted_sum / (
                self._preference_weight + self._resilience_weight + self._experience_weight)
        return scaled_weighted_sum

    def get_worker_allocation(self, filter_no_workers_assigned=False) -> list[dict]:
        allocations = []
        for row_idx, row in self._state.iterrows():

            allocated_workers = []

            allocated_worker_idxs = self._get_allocated_workers_in_row(row_idx)
            # map worker idx to worker id
            allocated_workers = [self._idx_to_worker_map[worker_idx] for worker_idx in allocated_worker_idxs]
            # remove worker_ prefix
            allocated_workers = [worker.replace('worker_', '') for worker in allocated_workers]

            allocation_element = {
                "Start": solver_time_to_timestamp(row['interval_start'], self._start_timestamp),
                "_Start_human_readable": self._human_readable_timestamp(
                    solver_time_to_timestamp(row['interval_start'], self._start_timestamp)),
                "Finish": solver_time_to_timestamp(row['interval_end'], self._start_timestamp),
                "_Finish_human_readable": self._human_readable_timestamp(
                    solver_time_to_timestamp(row['interval_end'], self._start_timestamp)),
                "Resource": row['line'],
                "Task": row['Task'],
                "geometry": row['geometry'],
                "order": row['Task'].split(' × ')[0],
                "required_workers": row['required_workers'],
                "workers": allocated_workers
            }
            if filter_no_workers_assigned and len(allocated_workers) == 0:
                continue

            allocations.append(allocation_element)

        return allocations


if __name__ == '__main__':

    # start_timestamp = 1693548000
    # step_1_output = cp_solver_output

    start_timestamp = 1693807200
    step_1_output = cp_solver_output2

    worker_availabilities = worker_availabilities
    geometry_line_mapping = geometry_line_mapping
    human_factor_data = human_factor

    env = CrfWorkerAllocationEnv(
        previous_step_output=step_1_output['allocations'],
        worker_availabilities=worker_availabilities,
        geometry_line_mapping=geometry_line_mapping,
        human_factor_data=human_factor_data,
        start_timestamp=start_timestamp,
        allocate_workers_on_the_same_line_if_possible=False,
    )

    env.render()
    env.reset()

    # from stable_baselines3.common.env_checker import check_env
    # check_env(env)

    terminal = False
    while not terminal:
        # print valid actions for all rows with is_current_interval == 1
        for row_idx, row in env.get_state().iterrows():
            if row['is_current_interval'] == 1:
                valid_actions = env.valid_action_tuples_for_row(row_idx)
                # log.critical(f"valid actions for row {row_idx}: {valid_actions}")

        action = np.random.choice(env.valid_action_list())
        log.info(f"action: {action} ({env.action_idx_to_action_tuple(action)})")
        _, rew, terminal, _, _ = env.step(action)
        log.info(f"reward: {rew}")
        env.render()

    log.info(pprint.pformat(env.get_worker_allocation()))
