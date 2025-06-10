import itertools
import pprint
from typing import Any

import sb3_contrib

import random
import gymnasium as gym
import pandas as pd
from gymcts.gymcts_agent import SoloMCTSAgent

from gymcts.gymcts_gym_env import SoloMCTSGymEnv
from jsp_vis.console import gantt_chart_console
from utils.crf_timestamp_solver_time_conversion import solver_time_to_timestamp, timestamp_to_solver_time, log_time

from demonstrator.crf_worker_allocation_gym_env import CrfWorkerAllocationEnv
from utils.logger import log

from demonstrator.crf_step_two_input import (cp_solver_output, cp_solver_output2, worker_availabilities,
                                             geometry_line_mapping, human_factor)


from demonstrator.crf_order_to_line_cp_solver import main as crf_solve_order_to_line_instance

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
        # return self.env.unwrapped.greedy_rollout_sparse()
        for _ in range(2):
            best_action = self.env.unwrapped.best_eager_action()
            if best_action is not None:
                self.env.step(best_action)

        return self.env.unwrapped.get_scaled_KPI_score()

    def get_state(self) -> pd.DataFrame:
        return self.env.unwrapped.get_state()


def _perform_order_to_line_mapping(
        api_payload: dict,
        start_time_timestamp: int,
        makespan_weight: int = 1,
        tardiness_weight: int = 1
) -> list[dict[str, int | Any]]:
    instance = api_payload

    log.info(f"received instance", extra=instance)
    log.info(pprint.pformat(instance["order_data"]))

    # static data
    human_factor = instance["human_factor"]
    availabilities = instance["availabilities"]
    geometry_line_mapping = instance["geometry_line_mapping"]

    # create a lookup table for geometry to lines and geometry to primary line
    geometry_to_lines_lookup_dict = {}
    # geometry_main_line_lookup_dict = {}

    for elem in geometry_line_mapping:
        geometry = elem["geometry"]
        main_line = elem["main_line"]
        lines = [main_line] + [
            line for line
            in elem["alternative_lines"]
            if line not in ["", "NA", "pomigl."]
        ]
        geometry_to_lines_lookup_dict[geometry] = lines
        # geometry_main_line_lookup_dict[geometry] = main_line

    def _get_lines_for_geometry(geometry: str):
        try:
            return geometry_to_lines_lookup_dict[geometry]
        except KeyError:
            log.warning(f"geometry {geometry} not present in 'geometry_line_mapping'. Ignoring this geometry.")
            return []

    def _calculate_duration(line: str, geometry: str, amount: int, molds: int) -> int:
        throughput_in_units_per_hour = None
        for elem in instance["throughput_mapping"]:
            if elem["line"] == line and elem["geometry"] == geometry:
                throughput_in_units_per_hour = elem["throughput"]
                break
        if throughput_in_units_per_hour is None:
            log.warning(
                f"throughput for line '{line}' and geometry '{geometry}' not present in 'throughput_mapping'."
                f"Defaulting to 300 units per hour.")
            throughput_in_units_per_hour = 300
        if throughput_in_units_per_hour == 0:
            log.warning(
                f"throughput for line '{line}' and geometry '{geometry}' is 0. This is most likely an error in the data. "
                f"Defaulting to 300 units per hour.")
            throughput_in_units_per_hour = 300

        if molds < 0:
            raise ValueError(f"molds must be >= 0. Got {molds}")

        setup_time_in_minutes = 15 * molds  # 15 minutes per mold
        throughput_in_units_per_minute = throughput_in_units_per_hour / 60
        duration_in_minutes = amount / throughput_in_units_per_minute + setup_time_in_minutes
        log.debug(
            f"setup time for producing '{geometry}' on line '{line}' with '{molds}' molds: {setup_time_in_minutes} minutes")
        log.debug(
            f"producing '{amount}' units of '{geometry}' on line '{line}' with a throughput of {throughput_in_units_per_hour} units per hour takes: {duration_in_minutes} minutes ({duration_in_minutes / 60:.2f} hours)")
        log.debug(
            f"the total duration for producing '{amount}' units of '{geometry}' on line '{line}' is: {duration_in_minutes} minutes ({duration_in_minutes / 60:.2f} hours)")
        return int(duration_in_minutes)

    # look up all possible lines to produce an order_data element
    temp = []
    for order_data_elem in instance["order_data"]:
        geometry = order_data_elem["geometry"]
        possible_lines = _get_lines_for_geometry(geometry)
        log.debug(f"geometry: {geometry} can be produced on lines: {possible_lines}")

        for line in possible_lines:
            temp.append(order_data_elem | {"line": line})

    # filter out lines not relevant lines (specified in the request)
    try:
        relevant_lines = instance["perform_allocation_for_lines"]
    except KeyError:
        relevant_lines = []

    if len(relevant_lines):
        log.debug(f"filtering out line not relevant lines. relevant lines:{relevant_lines}")
        temp = [elem for elem in temp if elem["line"] in relevant_lines]

    # calculate deadlines for each element
    temp = [
        elem | {
            "duration_in_minutes": _calculate_duration(
                line=elem["line"],
                geometry=elem["geometry"],
                amount=elem["amount"],
                molds=elem["mold"]
            )
        }
        for elem in temp
    ]

    # map deadline timestamps to solver time domain
    # start_time_timestamp = instance["start_time_timestamp"]
    log_time(start_time_timestamp, "start time: ")
    temp = [
        elem | {
            "deadline_solver_time": timestamp_to_solver_time(
                timestamp_to_convert=elem["deadline"],
                start_timestamp=start_time_timestamp
            )
        }
        for elem in temp
    ]

    # create Task field. Task is the pair of order and geometry
    # This is basically the 'order' for the solver. The cp solver assumes that there is only one geometry per order,
    # which turn out to not be the case. By introducing the 'Task' can omit rewriting the solver.
    for elem in temp:
        elem["Task"] = f"{elem['order']} × {elem['geometry']}"

    for elem in temp:
        if elem["geometry"] == "505597580" or elem["order"] == "SEV - 36":
            log.debug(pprint.pformat(elem))
        else:
            log.debug(pprint.pformat(elem))

    temp_grouped_by_task = {
        key: list(group)
        for key, group
        in itertools.groupby(temp, key=lambda elem: elem["Task"])
    }
    log.info(pprint.pformat(temp_grouped_by_task))

    # EXAMPLE_ORDER_INSTANCE = [
    #     # Order 0  [alternative 0, alternative 1, alternative 2
    #     [(30, 0, 1, 32), (27, 1, 1, 32)],  # alternative: (duration [h], line_id, priority, due_date [h])

    # creating mappings for the solver instance
    line_str_to_line_id_mapping = {
        line_str: idx
        for idx, line_str in enumerate(
            # use relevant lines if specified, otherwise use all lines
            relevant_lines if len(relevant_lines)
            # convert to set to remove duplicates,
            else list(set([elem["line"] for elem in temp]))  # todo: check if this is correct
        )
    }
    # invert the line_str_to_line_id_mapping for the lookup in the other direction
    line_id_mapping_to_line_str_mapping = {value: key for key, value in line_str_to_line_id_mapping.items()}

    task_idx_to_task_key_mapping = {}
    task_key_to_task_idx_mapping = {}

    solver_instance = []
    for task_idx, (task, alternatives_elem_list) in enumerate(temp_grouped_by_task.items()):

        task_idx_to_task_key_mapping[task_idx] = task
        task_key_to_task_idx_mapping[task] = task_idx

        alternatives = []

        for elem in alternatives_elem_list:
            duration = elem["duration_in_minutes"]
            deadline_solver_time = elem["deadline_solver_time"]
            line_of_this_alternative = line_str_to_line_id_mapping[elem["line"]]
            priority = str(elem["priority"]).lower() == "true"
            priority = int(priority)  # convert to 0 or 1

            # asser all data are integers
            assert isinstance(duration, int)
            assert isinstance(line_of_this_alternative, int)
            assert isinstance(deadline_solver_time, int)

            alternatives.append(
                (duration, line_of_this_alternative, priority, deadline_solver_time)
            )

        # append all alternatives for this task to the solver instance.
        solver_instance.append(alternatives)

    log.info(pprint.pformat(solver_instance))

    solution_dict = crf_solve_order_to_line_instance(
        makespan_weight=makespan_weight,
        tardiness_weight=tardiness_weight,
        order_list=solver_instance
    )

    try:
        log.info("creating gantt chart...")
        dict_for_gantt = [
            elem | {
                'Resource': f'Resource {elem["Resource"]}'
            } for elem in solution_dict
        ]
        gantt_chart_console(pd.DataFrame(dict_for_gantt),
                            n_machines=len(line_id_mapping_to_line_str_mapping.keys()), resource_naming='Resource')
        log.info(pprint.pformat(line_id_mapping_to_line_str_mapping))
    except Exception as e:
        log.warning(f"could not create gantt chart: {e}.")

    remapped_solution_dict = [
        {
            "Start": solver_time_to_timestamp(solver_time=elem["Start"], start_timestamp=start_time_timestamp),
            "Finish": solver_time_to_timestamp(solver_time=elem["Finish"], start_timestamp=start_time_timestamp),
            "Resource": line_id_mapping_to_line_str_mapping[elem["Resource"]],
            "Task": task_idx_to_task_key_mapping[elem["Task"]],
        }
        for elem in solution_dict
    ]

    # lookup geometry (don't retrieve it from the task name)
    for elem in remapped_solution_dict:
        task_key = elem["Task"]
        task_alternatives = temp_grouped_by_task[task_key]
        allocated_alternative = None
        for alternative in task_alternatives:
            if alternative["line"] == elem["Resource"]:
                allocated_alternative = alternative
                break
        else:
            log.warning(f"could not find geometry for task: {task_key}")

        # Note: here more information could be added to the output
        elem["geometry"] = allocated_alternative["geometry"]
        elem["order"] = allocated_alternative["order"]

    log.info(pprint.pformat(remapped_solution_dict))
    return remapped_solution_dict

def _get_setuptime_mapping(
        api_payload: dict,
) -> list[dict[str, int | Any]]:
    instance = api_payload

    # static data
    geometry_line_mapping = instance["geometry_line_mapping"]

    # create a lookup table for geometry to lines and geometry to primary line
    geometry_to_lines_lookup_dict = {}

    for elem in geometry_line_mapping:
        geometry = elem["geometry"]
        main_line = elem["main_line"]
        lines = [main_line] + [
            line for line
            in elem["alternative_lines"]
            if line not in ["", "NA", "pomigl."]
        ]
        geometry_to_lines_lookup_dict[geometry] = lines
        # geometry_main_line_lookup_dict[geometry] = main_line

    def _get_lines_for_geometry(geometry: str):
        try:
            return geometry_to_lines_lookup_dict[geometry]
        except KeyError:
            return []

    # look up all possible lines to produce an order_data element
    temp = []
    for order_data_elem in instance["order_data"]:
        geometry = order_data_elem["geometry"]
        possible_lines = _get_lines_for_geometry(geometry)
        for line in possible_lines:
            temp.append(order_data_elem | {"line": line})

    # filter out lines not relevant lines (specified in the request)
    try:
        relevant_lines = instance["perform_allocation_for_lines"]
    except KeyError:
        relevant_lines = []

    if len(relevant_lines):
        log.debug(f"filtering out line not relevant lines. relevant lines:{relevant_lines}")
        temp = [elem for elem in temp if elem["line"] in relevant_lines]


    # create Task field. Task is the pair of order and geometry
    # This is basically the 'order' for the solver. The cp solver assumes that there is only one geometry per order,
    # which turn out to not be the case. By introducing the 'Task' can omit rewriting the solver.
    set_uptime_map_in_seconds = {}
    for elem in temp:
        elem["Task"] = f"{elem['order']} × {elem['geometry']}"
        set_uptime_map_in_seconds[elem["Task"]] = elem["mold"] * 15 * 60 # 15 minutes per mold

    return set_uptime_map_in_seconds


def _get_task_total_amount_mapping(
    api_payload: dict,
) -> dict:
    instance = api_payload
    mapping = {}
    for order_data_elem in instance["order_data"]:
        geometry = order_data_elem["geometry"]
        order = order_data_elem["order"]
        amount = order_data_elem["amount"]
        mapping[f"{order} × {geometry}"] = amount
    return mapping



def solve_with_cp(env: CrfWorkerAllocationEnv, api_payload) -> dict:
    # do greedy rollout for now
    env.reset()
    env.greedy_rollout_sparse()

    allocation_with_worker_data = env.get_worker_allocation()
    experience, resilience, preference = env.get_KPIs()


    # add mold changes
    setup_time_mapping = _get_setuptime_mapping(api_payload)

    return {
        "experience": experience,
        "preference": preference,
        "resilience": resilience,
        "transparency": "high",
        "allocations": allocation_with_worker_data,
    }


def solve_with_rl(env: CrfWorkerAllocationEnv, focus='balance') -> dict:
    # placeholder for now until the function is implemented
    # do a greedy rollout for now
    if focus == 'balance':
        model = sb3_contrib.MaskablePPO.load(
            "crf_rl_model-action-4290_obs-(110, 241)_resilience-1_experience-1_preference-1.zip")
    elif focus == 'experience':
        model = sb3_contrib.MaskablePPO.load(
            "crf_rl_model-action-4290_obs-(110, 241)_resilience-0_experience-10_preference-0.zip")
    elif focus == 'resilience':
        model = sb3_contrib.MaskablePPO.load(
            "crf_rl_model-action-4290_obs-(110, 241)_resilience-10_experience-0_preference-0.zip")
    elif focus == 'preference':
        model = sb3_contrib.MaskablePPO.load(
            "crf_rl_model-action-4290_obs-(110, 241)_resilience-0_experience-0_preference-10.zip")
    else:
        raise ValueError(f"Unknown focus: {focus}")

    env.reset()
    done = False

    if env.action_space.n == model.action_space.n and env.observation_space.shape == model.observation_space.shape:
        obs, _ = env.reset()
        while not done:
            masks = env.valid_action_mask()
            action, _ = model.predict(observation=obs, deterministic=True, action_masks=masks)
            obs, rew, done, turn, info = env.step(action)
            log.info(f"Action: {action}, Reward: {rew}")
    else:
        env.greedy_rollout_sparse()

    allocation_with_worker_data = env.get_worker_allocation()
    experience, resilience, preference = env.get_KPIs()

    return {
        "experience": experience,
        "preference": preference,
        "resilience": resilience,
        "transparency": "low",
        "allocations": allocation_with_worker_data,
    }


def solve_with_mcts(env: CrfWorkerAllocationEnv, n_sim=2) -> dict:
    env.reset()
    env = CrfMCTSWrapper(env)

    agent = SoloMCTSAgent(
        env=env,
        clear_mcts_tree_after_step=False,
        render_tree_after_step=True,
        exclude_unvisited_nodes_from_render=True,
        number_of_simulations_per_step=n_sim,
    )
    actions = agent.solve(render_tree_after_step=True)

    experience, resilience, preference = env.unwrapped.get_KPIs()
    allocation_with_worker_data = env.unwrapped.get_worker_allocation()

    return {
        "experience": experience,
        "preference": preference,
        "resilience": resilience,
        "transparency": "medium",
        "allocations": allocation_with_worker_data,
    }


if __name__ == '__main__':
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
        preference_weight=0,
        resilience_weight=0,
        experience_weight=1,
    )

    # res = solve_with_cp(env)
    res = solve_with_rl(env, focus='experience')
    # res = solve_with_mcts(env, n_sim=2)

    log.info(pprint.pformat(res))
