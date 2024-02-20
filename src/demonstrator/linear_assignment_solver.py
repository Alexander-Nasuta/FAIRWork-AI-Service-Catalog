import pprint
import time
from utils.logger import log
from utils.project_paths import resources_dir_path
from validation.input_validation import validate_instance
from ortools.graph.python import linear_sum_assignment

from validation.output_valiation import validate_output_dict


def cost_function(
        Availability: str | None = None,
        MedicalCondition: str | None = None,
        UTEExperience: str | None = None,
        WorkerResilience: float | None = None,
        WorkerPreference: float | None = None,
        ProductionPriority: str | None = None,
        DueDate: int | None = None,
        dummy_task: bool = False,
) -> float | str:
    cost_supremum = 100_000  # cost for assigning a worker to a dummy task
    if dummy_task:
        return cost_supremum
    if Availability == "False":
        return "NA"
    if MedicalCondition == "False":
        return "NA"

    # calculate cost
    # this can be any function of the homogenized data and the order data
    # make sure that cost_supremum is bigger that the biggest possible cost

    experience_cost = 1 if UTEExperience == "False" else 10
    priority_cost = 1 if ProductionPriority == "True" else 10
    due_date_cost = DueDate
    preference_cost = 1 / (WorkerPreference + 0.01)
    resilience_cost = 1 / (WorkerResilience + 0.01)


    total_cost = 5 * preference_cost + \
                 1 * resilience_cost + \
                 1 * experience_cost + \
                 1 * priority_cost + \
                 1 * due_date_cost

    return total_cost


def allocate_using_linear_assignment_solver(instance: dict) -> dict:
    log.info("allocating using linear assignment solver")

    start_time_setup = time.perf_counter()

    order_info_list: list = instance["OrderInfoList"]
    num_orders = len(order_info_list)
    worker_ids_list: list = [worker_info["Id"] for worker_info in order_info_list[0]["WorkerInfoList"]]
    num_workers = len(worker_ids_list)

    # check number of required workers
    req_workers = sum([
        order_info["LineInfo"]["WorkersRequired"] for order_info in order_info_list
    ])

    log.info(f"num_orders: {num_orders}, num_workers: {num_workers}, "
             f"req_workers: {req_workers}")

    # construction matrix

    # row corresponds to a worker
    # column corresponds to an order
    # a cell is one "homogenized data" entry

    # cost matrix has to have the #workers rows and #workers colums
    tasks = [] # just for sanity check
    task_num_to_task_str = {}
    task_details = [] # used to pass values to cost function during matrix poulation

    for order_info_list_elem in order_info_list:
        for ith_worker_for_oder in range(order_info_list_elem["LineInfo"]["WorkersRequired"]):
            task_desc = f"Line_{order_info_list_elem['LineInfo']['LineId']}_worker_{ith_worker_for_oder}"
            task_num_to_task_str[len(tasks)] = task_desc
            tasks.append(task_desc)
            task_details.append({
                "task": task_desc,
                "line_info": order_info_list_elem["LineInfo"],
                "worker": f"worker_{ith_worker_for_oder}",
                "is_dummy": False,
            })

    required_dummy_tasks = num_workers - req_workers
    for dummy_task in range(required_dummy_tasks):
        task_num_to_task_str[len(tasks)] = f"dummy_task_{dummy_task}"
        tasks.append(f"not_assigned_worker_{dummy_task}")
        task_details.append(
            {
                "task": f"not_assigned_worker_{dummy_task}",
                "line_info": None,
                "worker": None,
                "is_dummy": True,
            }
        )
    log.info(f"tasks: {tasks[:req_workers]}")
    log.info(f"number of dummy tasks: {required_dummy_tasks} "
             f"( = {num_workers} (#wokers)- {req_workers} (#required workers) )")

    log.info("creating empty cost matrix...")
    cost_matrix = [
        ['NA' for _ in range(num_workers)]
        for _ in range(num_workers)
    ]
    log.info("populating cost matrix...")
    for w_row_idx, worker_id in enumerate(worker_ids_list):
        for t_col_idx, td in enumerate(task_details):#

            if td["is_dummy"]:
                cost_matrix[w_row_idx][t_col_idx] = cost_function(dummy_task=True)
                continue
            line_info = td["line_info"]

            # find matchin order info list element

            order_info_elem = None
            for order_info_elem in order_info_list:
                if order_info_elem["LineInfo"]["LineId"] == line_info["LineId"]:
                    # find matching worker info list element
                    worker_info = None
                    for worker_info_elem in order_info_elem["WorkerInfoList"]:
                        if worker_info_elem["Id"] == worker_id:
                            cost_fn_data = {
                                "Availability": worker_info_elem["Availability"],
                                "MedicalCondition": worker_info_elem["MedicalCondition"],
                                "UTEExperience": worker_info_elem["UTEExperience"],
                                "WorkerResilience": worker_info_elem["WorkerResilience"],
                                "WorkerPreference": sum([
                                    e["Value"]
                                    for e in worker_info_elem["WorkerPreference"]
                                    if e["LineId"] == line_info["LineId"]
                                ]),
                                "dummy_task": False,
                                "ProductionPriority": line_info["ProductionPriority"],
                                "DueDate": line_info["DueDate"],
                            }
                            cost_matrix[w_row_idx][t_col_idx] = cost_function(**cost_fn_data)
                            break
                    else: # else block is executed if the loop did not break
                        raise ValueError(f"worker_id not found in order_info_list")
                    break
            else: # else block is executed if the loop did not break
                raise ValueError(f"line_info['LineId'] not found in order_info_list")

    end_time_setup = time.perf_counter()

    start_time_solver_setup = time.perf_counter()
    assignment = linear_sum_assignment.SimpleLinearSumAssignment()
    for worker in range(0, num_workers):
        for task in range(0, num_workers):
            if cost_matrix[worker][task] != 'NA':
                assignment.add_arcs_with_cost(worker, task, cost_matrix[worker][task])

    end_time_solver_setup = time.perf_counter()

    solver_setup_duration = end_time_solver_setup - start_time_solver_setup

    start_time_solving = time.perf_counter()
    status = assignment.solve()
    end_time_solving = time.perf_counter()

    solving_duration = end_time_solving - start_time_solving

    res = {
        e["LineInfo"]["LineId"]: {
            "LineId": e["LineInfo"]["LineId"],
            "WorkersRequired": e["LineInfo"]["WorkersRequired"],
            "Workers": [],
        }
        for e in order_info_list
    }
    if status == assignment.OPTIMAL:
        total_cost = 0
        for i in range(0, assignment.num_nodes()):
            t_index = assignment.right_mate(i)
            t_details = task_details[t_index]

            if t_details['is_dummy']:
                continue

            t_des = tasks[t_index]

            # find matching order info list element
            order_info_elem = None
            for order_info_elem in order_info_list:
                if order_info_elem["LineInfo"]["LineId"] == t_details["line_info"]["LineId"]:
                    # find matching worker info list element
                    worker_info = None
                    for worker_info_elem in order_info_elem["WorkerInfoList"]:
                        if worker_info_elem["Id"] == worker_ids_list[i]:
                            res[t_details["line_info"]["LineId"]]["Workers"].append(
                                {
                                    "Id": worker_info_elem["Id"],
                                    "Availability": worker_info_elem["Availability"],
                                    "MedicalCondition":  worker_info_elem["MedicalCondition"],
                                    "UTEExperience": worker_info_elem["UTEExperience"],
                                    "WorkerResilience": worker_info_elem["WorkerResilience"],
                                    # fin matching worker preference
                                    "WorkerPreference": sum([ # using sum instead of using index 0, just coding style
                                        e["Value"]
                                        for e in worker_info_elem["WorkerPreference"]
                                        if e["LineId"] == t_details["line_info"]["LineId"]
                                    ]),
                                }
                            )
                            break
        res = {
            "AllocationList": [
                val for line_id, val in res.items()
            ]
        }
        log.info(f"'OPTIMAL' solution found: {pprint.pformat(res)}", extra=res)

        validate_output_dict(res)

        return res

    raise ValueError(f"no optimal solution found")


if __name__ == '__main__':
    import json

    data = None
    # file = resources_dir_path.joinpath("OutputKBv2.json")
    # file = resources_dir_path.joinpath("OutputKB.json")
    file = resources_dir_path.joinpath("OutputKB_Final.json")
    with open(file) as json_file:
        data = json.load(json_file)

    validate_instance(data)

    allocate_using_linear_assignment_solver(data)
