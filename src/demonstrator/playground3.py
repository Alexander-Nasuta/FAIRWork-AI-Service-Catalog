import pprint
from datetime import datetime
from collections import namedtuple
from typing import Any

from demonstrator.crf_step_two_input import cp_solver_output2, cp_solver_output_mini, worker_availabilities
from demonstrator.crf_step_two_input import geometry_line_mapping as geo_mapping
from utils.crf_timestamp_solver_time_conversion import timestamp_to_solver_time, solver_time_to_timestamp

import pytz

timezone = pytz.FixedOffset(0)

line_allocations = cp_solver_output2['allocations']
worker_availabilities = worker_availabilities
my_geometry_line_mapping = geo_mapping
start_timestamp = 1693807200

# mapt line allocation to solver time domain
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


if __name__ == '__main__':
    # print(pprint.pformat(worker_availabilities))
    # print(pprint.pformat(line_allocations))
    def print_human_readable_timestamp(timestamp) -> str:
        human_readable_time = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
        return human_readable_time.strftime('%A %Y-%m-%d %H:%M:%S')


    n_intervals = len(interval_tuple)
    print(f"the schedule is devided into {n_intervals} intervals: {interval_tuple}")


    def intervals_overlap(interval1, interval2):
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
            f"interval {interval_idx}: {print_human_readable_timestamp(solver_time_to_timestamp(interval_start, start_timestamp))} - {print_human_readable_timestamp(solver_time_to_timestamp(interval_end, start_timestamp))}")
        print("#" * 80)
        # find line allocations that are within this interval
        line_allocations_within_interval = [
            elem for elem in line_allocations
            if
            intervals_overlap((elem['Start_solver_time'], elem['Finish_solver_time']), (interval_start, interval_end))
        ]
        print("line allocations within this interval:")
        print(pprint.pformat(line_allocations_within_interval))
        print("available wrorkers within this interval:")
        workers_within_interval = [
            elem for elem in worker_availabilities
            if intervals_overlap((elem['from_solver_time'], elem['end_solver_time']), (interval_start, interval_end))
        ]
        print(pprint.pformat(workers_within_interval))

        if len(workers_within_interval) and len(line_allocations_within_interval):
            relevant_intervals.append((interval_start, interval_end))

    print(f"Relevant intervals: {relevant_intervals}")


    def get_required_number_of_workers(line: str, geometry: str, geometry_line_mapping: list[dict[str, Any]]) -> int:
        for elem in geometry_line_mapping:
            if elem['geometry'] == geometry:
                if elem['main_line'] == line or line in elem['alternative_lines']:
                    return elem['number_of_workers']
        else:
            print(f"no line allocation found for line {line} and geo {geometry}. Returning a default value of 4")
            return 4


    data = []
    for interval_idx, (interval_start, interval_end) in enumerate(relevant_intervals):
        line_allocations_within_interval = [
            elem for elem in line_allocations
            if
            intervals_overlap((elem['Start_solver_time'], elem['Finish_solver_time']), (interval_start, interval_end))
        ]
        workers_within_interval = [
            elem for elem in worker_availabilities
            if intervals_overlap((elem['from_solver_time'], elem['end_solver_time']), (interval_start, interval_end))
        ]

        for line_elem in line_allocations_within_interval:
            worker_decision_variables = namedtuple(
                'WorkerVars',
                ['available', 'medical_condition', 'preference', 'resilience', 'experience', 'allocated']
            )

            required_workers = get_required_number_of_workers(
                line=line_elem['Resource'],
                geometry=line_elem['geometry'],
                geometry_line_mapping=my_geometry_line_mapping
            )
            print(f"required workers for line '{line_elem['Resource']}' and geometry '{line_elem['geometry']}' is {required_workers}")

            data_row_dict = {
                'interval_no': interval_idx,
                'is_current_interval': 0,
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
                intervals_overlap((elem['from_solver_time'], elem['end_solver_time']), (interval_start, interval_end))
            ]
            for worker_in_interval in workers_within_interval:

                worker_id = worker_in_interval['worker']

                res = worker_decision_variables(
                    available=1,
                    medical_condition=0,
                    preference=0,
                    resilience=0,
                    experience=0,
                    allocated=0
                )
                data_row_dict = data_row_dict | {
                    f'worker_{worker_id}': res
                }
                print(data_row_dict)
            print(pprint.pformat(data_row_dict))
