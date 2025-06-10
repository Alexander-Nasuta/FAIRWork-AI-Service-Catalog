import pprint

from demonstrator.algo_collection import _perform_order_to_line_mapping, solve_with_cp
from demonstrator.crf_worker_allocation_gym_env import CrfWorkerAllocationEnv
from utils.project_paths import resources_dir_path


def no_availabilites_on_start_day_values():
    import json

    api_data = None
    # file = resources_dir_path.joinpath("OutputKBv2.json")
    # file = resources_dir_path.joinpath("OutputKB.json")
    file = resources_dir_path.joinpath("test_cases_ai_service").joinpath("input_no_availability_data_on_start_date_500.json")
    with open(file) as json_file:
        api_data = json.load(json_file)

    start_timestamp = api_data["start_time_timestamp"]

    worker_availabilities = api_data["availabilities"]
    geometry_line_mapping = api_data["geometry_line_mapping"]
    human_factor_data = api_data["human_factor"]
    order_data = api_data["order_data"]
    throughput_mapping = api_data["throughput_mapping"]

    earliest_worker_availability = min(
        [elem['from_timestamp'] for elem in worker_availabilities],
    )

    start_timestamp = max(start_timestamp, earliest_worker_availability)

    allocations_dict = _perform_order_to_line_mapping(
        api_payload=api_data,
        start_time_timestamp=start_timestamp,
        makespan_weight=1,
        tardiness_weight=1
    )



    env = CrfWorkerAllocationEnv(
        previous_step_output=allocations_dict,
        worker_availabilities=worker_availabilities,
        geometry_line_mapping=geometry_line_mapping,
        human_factor_data=human_factor_data,
        start_timestamp=start_timestamp,
        allocate_workers_on_the_same_line_if_possible=False,
        preference_weight=1,
        resilience_weight=1,
        experience_weight=1,

        order_data=order_data,
        throughput_mapping=throughput_mapping,
    )

    res = solve_with_cp(env, api_payload=api_data)
    print(pprint.pformat(res))
    return res


if __name__ == '__main__':
    no_availabilites_on_start_day_values()
