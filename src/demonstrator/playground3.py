import pprint
from datetime import datetime
from collections import namedtuple
from typing import Any

from demonstrator.crf_step_two_input import cp_solver_output2, cp_solver_output_mini, worker_availabilities, human_factor
from demonstrator.crf_step_two_input import geometry_line_mapping as geo_mapping
from utils.crf_timestamp_solver_time_conversion import timestamp_to_solver_time, solver_time_to_timestamp

import pytz
import pandas as pd

timezone = pytz.FixedOffset(0)

line_allocations = cp_solver_output2['allocations']
worker_availabilities = worker_availabilities
my_geometry_line_mapping = geo_mapping
human_factor_data = human_factor
start_timestamp = 1693807200

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

interval_bounds = set()

for line_allocation_elem in line_allocations:
    interval_bounds.add(line_allocation_elem['Start_solver_time'])
    interval_bounds.add(line_allocation_elem['Finish_solver_time'])

for worker_availabilities_elem in worker_availabilities:
    interval_bounds.add(worker_availabilities_elem['from_solver_time'])
    interval_bounds.add(worker_availabilities_elem['end_solver_time'])

interval_bounds_ascending_list = sorted(list(interval_bounds))

interval_tuple = []  # [(start, end), ...]
for interval_start, interval_end in zip(interval_bounds_ascending_list[:-1], interval_bounds_ascending_list[1:]):
    interval_tuple.append((interval_start, interval_end))


# getters for human factor data:
#            medical_condition=0,
#                     preference=0,
#                     resilience=0,
#                     experience=0,
#  {
#             "geometry": "533908540",
#             "preference": 0.22,
#             "resilience": 0.73,
#             "medical_condition": "true",
#             "experience": 0.33,
#             "worker": "15004479"
# },
def _get_human_factor_data(worker:str, geometry:str, human_factor_data:dict[str, Any]) -> (float, float, bool, float):
    for elem in human_factor_data:
        if elem['geometry'] == geometry and elem['worker'] == worker:
            preference = elem['preference']
            resilience = elem['resilience']
            medical_condition = elem['medical_condition'].lower() == "true"
            experience = elem['experience']
            return preference, resilience, medical_condition, experience
    else:
        print(f"no human factor data found for worker {worker} and geometry {geometry}. Returning default values")
        return 0, 0, False, 0


if __name__ == '__main__':
    # print(pprint.pformat(worker_availabilities))
    # print(pprint.pformat(line_allocations))
    def _print_human_readable_timestamp(timestamp) -> str:
        human_readable_time = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
        return human_readable_time.strftime('%A %Y-%m-%d %H:%M:%S')


    n_intervals = len(interval_tuple)
    print(f"the schedule is devided into {n_intervals} intervals: {interval_tuple}")


    def _intervals_overlap(interval1, interval2):
        start1, end1 = interval1
        start2, end2 = interval2
        return start1 < end2 and start2 < end1


    relevant_intervals = []

    for interval_idx, (interval_start, interval_end) in enumerate(interval_tuple):
        print("#" * 80)
        print("#" * 80)
        print(
            f"interval {interval_idx}: {interval_start} - {interval_end} ({solver_time_to_timestamp(interval_start, start_timestamp)}-{solver_time_to_timestamp(interval_end, start_timestamp)})")
        print(
            f"interval {interval_idx}: {_print_human_readable_timestamp(solver_time_to_timestamp(interval_start, start_timestamp))} - {_print_human_readable_timestamp(solver_time_to_timestamp(interval_end, start_timestamp))}")
        print("#" * 80)
        # find line allocations that are within this interval
        line_allocations_within_interval = [
            elem for elem in line_allocations
            if
            _intervals_overlap((elem['Start_solver_time'], elem['Finish_solver_time']), (interval_start, interval_end))
        ]
        print("line allocations within this interval:")
        print(pprint.pformat(line_allocations_within_interval))
        print("available wrorkers within this interval:")
        workers_within_interval = [
            elem for elem in worker_availabilities
            if _intervals_overlap((elem['from_solver_time'], elem['end_solver_time']), (interval_start, interval_end))
        ]
        print(pprint.pformat(workers_within_interval))

        if len(workers_within_interval) and len(line_allocations_within_interval):
            relevant_intervals.append((interval_start, interval_end))

    print(f"Relevant intervals: {relevant_intervals}")


    def _get_required_number_of_workers(line: str, geometry: str, geometry_line_mapping: list[dict[str, Any]]) -> int:
        for elem in geometry_line_mapping:
            if elem['geometry'] == geometry:
                if elem['main_line'] == line or line in elem['alternative_lines']:
                    return elem['number_of_workers']
        else:
            print(f"no line allocation found for line {line} and geo {geometry}. Returning a default value of 4")
            return 4


    df_data = []
    for interval_idx, (interval_start, interval_end) in enumerate(relevant_intervals):
        line_allocations_within_interval = [
            elem for elem in line_allocations
            if
            _intervals_overlap((elem['Start_solver_time'], elem['Finish_solver_time']), (interval_start, interval_end))
        ]
        workers_within_interval = [
            elem for elem in worker_availabilities
            if _intervals_overlap((elem['from_solver_time'], elem['end_solver_time']), (interval_start, interval_end))
        ]

        for line_elem in line_allocations_within_interval:
            worker_decision_variables = namedtuple(
                'WorkerVars',
                ['available', 'medical_condition', 'preference', 'resilience', 'experience', 'allocated']
            )

            required_workers = _get_required_number_of_workers(
                line=line_elem['Resource'],
                geometry=line_elem['geometry'],
                geometry_line_mapping=my_geometry_line_mapping
            )
            print(f"required workers for line '{line_elem['Resource']}' and geometry '{line_elem['geometry']}' is {required_workers}")

            data_row_dict = {
                'interval_no': interval_idx,
                'is_current_interval': 1 if interval_idx == 0 else 0,
                'interval_start': interval_start,
                'interval_end': interval_end,
                'Task': line_elem['Task'],
                'line': line_elem['Resource'],
                'geometry': line_elem['geometry'],
                'required_workers': required_workers,
                'allocated_workers': 0,
                'required_workers_met': 0,
            }

            workers_within_interval = [
                elem for elem in worker_availabilities
                if
                _intervals_overlap((elem['from_solver_time'], elem['end_solver_time']), (interval_start, interval_end))
            ]
            n_workers_available = 0
            for worker_in_interval in workers_within_interval:

                worker_id = worker_in_interval['worker']
                preference, resilience, medical_condition, experience = _get_human_factor_data(
                    worker=worker_id,
                    geometry=line_elem['geometry'],
                    human_factor_data=human_factor_data
                )

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
    print(df["worker_15016633"].head(n=25))
