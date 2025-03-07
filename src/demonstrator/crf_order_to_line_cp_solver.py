from typing import List, Any

import pandas
import pandas as pd

from utils.logger import log

import collections

from jsp_vis.console import gantt_chart_console
from ortools.sat.python import cp_model

EXAMPLE_ORDER_INSTANCE = [
    # Order 0  [alternative 0, alternative 1, alternative 2
    [(30, 0, 1, 32), (27, 1, 1, 32)],  # alternative: (duration [h], line_id, priority, due_date [h])
    # Order 1
    [(20, 0, 0, 40), (30, 1, 0, 40)],
    # Order 2
    [(20, 0, 0, 48), (30, 1, 0, 48), (15, 2, 0, 48)],
    # Order 3
    [(20, 0, 0, 40), (25, 2, 0, 40)],
    # Order 4
    [(10, 0, 1, 16)],
    # Order 5
    [(20, 0, 0, 40), (10, 1, 0, 40)],
    # Order 6
    [(6, 0, 0, 16), (7, 2, 0, 16)],
    # Order 7
    [(5, 2, 0, 16)],
    # Order 8
    [(7, 1, 1, 8), (8, 2, 1, 8)],
    # Order 9
    [(16, 0, 0, 24), (13, 2, 0, 24)],
    # Order 10
    [(12, 0, 0, 32), (10, 1, 0, 32)],
    # Order 11
    [(7, 0, 0, 80), (8, 2, 0, 80)],
    # Order 12
    [(5, 1, 0, 80)],
    # Order 13
    [(6, 0, 0, 80)],
    # Order 14
    [(4, 2, 0, 80)],
    # Order 14
    [(4, 2, 0, 80)],
    # Order 15
    [(6, 1, 0, 72)],
    # Order 16
    [(13, 1, 0, 60), (16, 2, 0, 60)],
    # Order 17
    [(4, 1, 0, 60), (8, 2, 0, 60)],
    # Order 18
    [(4, 0, 0, 60), (6, 1, 0, 60)],
    # Order 19
    [(4, 0, 0, 60), (6, 1, 0, 60)],
    # Order 20
    [(5, 0, 0, 60), (7, 2, 0, 60)],
    # Order 21
    [(8, 1, 0, 60), (9, 2, 0, 60)],
    # Order 23
    [(7, 0, 0, 60), (8, 1, 0, 60)],
    # Order 24
    [(3, 0, 0, 60), (4, 2, 0, 60)],
    # Order 25
    [(4, 0, 0, 80), (4, 2, 0, 80)],
]

EXAMPLE_ORDER_INSTANCE_2 = [[(1348, 2, 0, 7200), (1348, 1, 0, 7200)],
             [(690, 1, 0, 7200)],
             [(675, 0, 0, 7200)],
             [(635, 0, 0, 7200)],
             [(690, 1, 0, 7200)],
             [(1390, 2, 0, 7200), (1390, 1, 0, 7200)],
             [(1290, 2, 0, 7200), (1290, 1, 0, 7200)],
             [(1320, 1, 0, 7200)],
             [(435, 2, 0, 7200), (435, 1, 0, 7200), (435, 0, 0, 7200)],
             [(635, 0, 0, 14400)],
             [(675, 0, 0, 14400)],
             [(690, 1, 0, 14400)],
             [(1348, 2, 0, 14400), (1348, 1, 0, 14400)],
             [(1390, 2, 0, 14400), (1390, 1, 0, 14400)],
             [(1290, 2, 0, 14400), (1290, 1, 0, 14400)],
             [(235, 2, 0, 14400)],
             [(675, 0, 0, 21600)],
             [(635, 0, 0, 21600)],
             [(690, 1, 0, 21600)],
             [(690, 1, 0, 21600)],
             [(1320, 1, 0, 21600)],
             [(456, 1, 0, 21600)],
             [(575, 2, 0, 21600)],
             [(1390, 2, 0, 21600), (1390, 1, 0, 21600)],
             [(1290, 2, 0, 21600), (1290, 1, 0, 21600)]]


class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0

    def on_solution_callback(self):
        """Called at each new solution."""
        log.debug(f"Solution {self.__solution_count}, time = {self.wall_time:.2f} s, cost = {self.objective_value:.2f}")
        self.__solution_count += 1


def main(makespan_weight: int = 1, tardiness_weight: int = 1, hours_per_day: int = 16,
         order_list: List[Any] = None) -> (pandas.DataFrame, dict):
    log.info("Running main function")
    log.info(f"makespan_weight: {makespan_weight}, tardiness_weight: {tardiness_weight}")

    # Model the crf line allocation problem.
    model = cp_model.CpModel()

    # 'orders' can be calculated from data in the database
    # using EXAMPLE_ORDER_INSTANCE for now
    orders = order_list

    # calculate horizon
    # for each oder add up the duration of the longest alternative
    horizon = sum(
        max((task_tuple[0] for task_tuple in order_alternatives_list), default=0)  # index 0 is the duration
        for order_alternatives_list in orders
    )
    log.debug(f"Horizon = {horizon}")

    line_set = set()
    for order in orders:
        for alternative in order:
            line_set.add(alternative[1])
    n_machines = len(line_set)  # used for gantt chart

    # Global storage of variables.
    intervals_per_resources = collections.defaultdict(list)
    starts_order_line = {}  # indexed by (order_idx, line_id).
    starts_order = {}  # indexed by (order_idx).
    tradiness = {}  # indexed by (order_idx).
    presences = {}  # indexed by (order_idx, alternative_idx).
    order_ends = {}  # indexed by (order_idx).

    priority_starts = []
    non_priority_starts = []

    # create alternative allocation intervals for each order
    for order_idx, order in enumerate(orders):

        order_line_allocation_presences = []

        order_start = model.new_int_var(0, horizon, f"order_{order_idx}_start")
        order_end = model.new_int_var(0, horizon, f"order_{order_idx}_end")

        order_ends[order_idx] = order_end
        starts_order[order_idx] = order_start

        # handle priority and non-priority starts
        order_is_priority = order[0][2] == 1  # index 2 is the priority
        if order_is_priority:
            priority_starts.append(order_start)
        else:
            non_priority_starts.append(order_start)

        order_tardiness = model.new_int_var(0, horizon, f"order_{order_idx}_tardiness")
        tradiness[order_idx] = order_tardiness

        for alternative in order:
            # unpack alternative tuple
            _duration, _line_id, _priority, _due_date = alternative

            suffix_str = f"order_{order_idx}_line_{_line_id}"
            alt_presence = model.new_bool_var(f"presence_{suffix_str}")
            alt_start = model.new_int_var(0, horizon, f"start_{suffix_str}")
            alt_end = model.new_int_var(0, horizon, f"end_{suffix_str}")

            alt_interval = model.new_optional_interval_var(
                alt_start, _duration, alt_end, alt_presence, f"interval_{suffix_str}"
            )

            alt_task_tardiness = model.new_int_var(0, horizon, f"tardiness_{suffix_str}")
            # Add a constraint that max_tardiness is equal to the maximum of 0 and (alt_end - _due_date)
            model.add_max_equality(alt_task_tardiness, [0, alt_end - _due_date])

            # set the end of the order to the end of the selected alternative
            model.add(order_end == alt_end).only_enforce_if(alt_presence)
            model.add(order_start == alt_start).only_enforce_if(alt_presence)
            model.add(order_tardiness == alt_task_tardiness).only_enforce_if(alt_presence)

            # Add the local interval to the right machine.
            intervals_per_resources[_line_id].append(alt_interval)

            # Add to presences
            presences[(order_idx, _line_id)] = alt_presence
            starts_order_line[(order_idx, _line_id)] = alt_start

            # group all alternative presences for an order in a list
            # to enforce only one alternative after the loop
            order_line_allocation_presences.append(alt_presence)

        # only one alternative/ one allocation can be selected
        model.add_exactly_one(order_line_allocation_presences)

    # line allocation constraints
    # a line can only process one order at a time
    for line_id, line_intervals in intervals_per_resources.items():
        model.add_no_overlap(line_intervals)

    # todo: do the priority constrain machine/line wise

    # priority starts before non-priority constraints
    for priority_start in priority_starts:
        for non_priority_start in non_priority_starts:
            model.add(priority_start <= non_priority_start)

    # objective function
    # the objective function is to minimize the makespan and the tardiness weighted 1:1

    # makespan = end time of the last order
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, [oe for oe in order_ends.values()])

    # total tardiness
    total_tardiness = model.new_int_var(0, horizon * len(orders), "total_tardiness")
    # sum up tardiness of all orders
    model.add(total_tardiness == sum(tradiness[order_idx] for order_idx in range(len(orders))))

    # defining a variable for the objective function
    # minimize makespan and tardiness weighted by 1:1 by default
    objective_var = model.new_int_var(0, horizon * len(orders), "objective")
    model.add(objective_var == makespan_weight * makespan + tardiness_weight * total_tardiness)
    log.info(f"Cost function: cost = {makespan_weight} * makespan + {tardiness_weight} * total_tardiness")

    model.minimize(objective_var)

    # Solve model.
    solver = cp_model.CpSolver()
    solution_printer = SolutionPrinter()
    status = solver.solve(model, solution_printer)

    log.info(f"""
Solution found: {status == cp_model.OPTIMAL or status == cp_model.FEASIBLE}
Solution is optimal: {status == cp_model.OPTIMAL}

Makespan: {solver.Value(makespan)} minutes
Total Tardiness: {solver.Value(total_tardiness)} minutes (sum of all tardiness values of the orders)
Cost: {solver.Value(objective_var)} (measures the quality of the solution based on the given weights of the cost function)
    """)

    makespan = solver.Value(makespan)

    log.info(f"Gantt chart (time window: 0-{makespan})")
    gantt_data = []
    for orders_idx, order_alternatives_list in enumerate(orders):
        try:
            line_idx = [
                line_id for _, line_id, *_ in order_alternatives_list
                if solver.Value(presences[(orders_idx, line_id)])
            ][0]
        except IndexError:
            # Fallback to 0 if the list is empty
            line_idx = next((line_id for _, line_id, *_ in order_alternatives_list), 0)
        gantt_data.append({
            'Task': orders_idx,
            'Start': solver.Value(starts_order[orders_idx]),
            'Finish': solver.Value(order_ends[orders_idx]),
            'Resource': line_idx
        })

    return gantt_data


if __name__ == '__main__':
    solution_dict = main(order_list=EXAMPLE_ORDER_INSTANCE_2)
    dict_for_gantt = [
        elem | {
            'Resource': f'Line {elem["Resource"]}'
        } for elem in solution_dict
    ]
    gantt_chart_console(pd.DataFrame(dict_for_gantt), n_machines=3, resource_naming='Line')
