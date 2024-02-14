from numbers import Real

from utils.logger import log
from utils.project_paths import resources_dir_path


def validate_instance(instance: dict) -> None:
    expected_key_types = [
        ("OrderInfoList", list),
    ]
    for key, expected_type in expected_key_types:
        if key not in instance:
            log.error(f"instance missing key '{key}'")
            raise KeyError(f"instance missing key '{key}'")
        if not isinstance(instance[key], expected_type):
            log.error(f"instance key '{key}' has unexpected type '{type(instance[key])}'")
            raise TypeError(f"instance key '{key}' has unexpected type '{type(instance[key])}'")

    validate_order_info_list(instance["OrderInfoList"])


def validate_order_info_list(order_info_list: list) -> None:
    if not isinstance(order_info_list, list):
        log.error(f"order_info_list has unexpected type '{type(order_info_list)}'")
        raise TypeError(f"order_info_list has unexpected type '{type(order_info_list)}'")

    for order in order_info_list:
        validate_order_info_list_elem(order)


def validate_order_info_list_elem(order_info_list_elem: dict) -> None:
    expected_key_types = [
        ("WorkerInfoList", list),
        ("LineInfo", dict),
    ]
    for key, expected_type in expected_key_types:
        if key not in order_info_list_elem:
            log.error(f"order_info_list_elem missing key '{key}'")
            raise KeyError(f"order_info_list_elem missing key '{key}'")
        if not isinstance(order_info_list_elem[key], expected_type):
            log.error(f"order_info_list_elem key '{key}' has unexpected type '{type(order_info_list_elem[key])}'")
            raise TypeError(f"order_info_list_elem key '{key}' has unexpected type '{type(order_info_list_elem[key])}'")

    # worker info list has to have length 159
    if len(order_info_list_elem["WorkerInfoList"]) != 159:
        log.error(f"worker_info_list has unexpected length '{len(order_info_list_elem['WorkerInfoList'])}'")
        raise ValueError(f"worker_info_list has unexpected length '{len(order_info_list_elem['WorkerInfoList'])}'")

    # validate line info
    validate_line_info(order_info_list_elem["LineInfo"])

    for elem in order_info_list_elem["WorkerInfoList"]:
        validate_worker_info_list_elem(elem)

    # check if all workers are unique
    worker_ids = [elem["Id"] for elem in order_info_list_elem["WorkerInfoList"]]
    if len(worker_ids) != len(set(worker_ids)):
        log.error(f"worker_info_list contains duplicate worker ids")
        raise ValueError(f"worker_info_list contains duplicate worker ids")



def validate_worker_info_list_elem(info_list_elem: dict) -> None:
    expected_key_types = [
        ("Id", str),
        ("Availability", str),
        ("MedicalCondition", str),
        ("UTEExperience", str),
        ("WorkerResilience", float),
        ("WorkerPreference", list),
    ]
    for key, expected_type in expected_key_types:
        if key not in info_list_elem:
            log.error(f"info_list_elem missing key '{key}'")
            raise KeyError(f"info_list_elem missing key '{key}'")
        if not isinstance(info_list_elem[key], expected_type):
            log.error(f"info_list_elem key '{key}' has unexpected type '{type(info_list_elem[key])}'")
            raise TypeError(f"info_list_elem key '{key}' has unexpected type '{type(info_list_elem[key])}'")

        if not 0 <= info_list_elem["WorkerResilience"] <= 1:
            log.error(f"WorkerResilience has unexpected value '{info_list_elem['WorkerResilience']}'")
            raise ValueError(f"WorkerResilience has unexpected value '{info_list_elem['WorkerResilience']}'. "
                             f"Expected value between 0 and 1.")


def validate_worker_preference_list(worker_preference_list: list) -> None:
    if not isinstance(worker_preference_list, list):
        log.error(f"worker_preference_list has unexpected type '{type(worker_preference_list)}'")
        raise TypeError(f"worker_preference_list has unexpected type '{type(worker_preference_list)}'")

    def validate_worker_preference_elem_types(worker_preference: dict) -> None:
        expected_key_types = [
            ("LineId", str),
            ("Value", Real),
        ]
        for key, expected_type in expected_key_types:
            if key not in worker_preference:
                log.error(f"worker_preference missing key '{key}'")
                raise KeyError(f"worker_preference missing key '{key}'")
            if not isinstance(worker_preference[key], expected_type):
                log.error(f"worker_preference key '{key}' has unexpected type '{type(worker_preference[key])}'")
                raise TypeError(f"worker_preference key '{key}' has unexpected type '{type(worker_preference[key])}'")

        if not 0 <= worker_preference["Value"] <= 1:
            log.error(f"worker_preference has unexpected value '{worker_preference['Value']}'")
            raise ValueError(f"worker_preference has unexpected value '{worker_preference['Value']}'. "
                             f"Expected value between 0 and 1.")

    for worker_preference in worker_preference_list:
        validate_worker_preference_elem_types(worker_preference)

    # check if all preferences add up to 1
    eps = 1e-6
    if abs(sum([pref["Value"] for pref in worker_preference_list]) - 1) > eps:
        log.error(f"worker_preference_list does not add up to 1")
        raise ValueError(f"worker_preference_list does not add up to 1")


def validate_line_info(line_info: dict) -> None:
    expected_key_types = [
        ("LineId", str),
        ("Geometry", str),
        ("ProductionPriority", str),
        ("DueDate", int),
        ("WorkersRequired", int)
    ]
    for key, expected_type in expected_key_types:
        if key not in line_info:
            log.error(f"line_info missing key '{key}'")
            raise KeyError(f"line_info missing key '{key}'")
        if not isinstance(line_info[key], expected_type):
            log.error(f"line_info key '{key}' has unexpected type '{type(line_info[key])}'")
            raise TypeError(f"line_info key '{key}' has unexpected type '{type(line_info[key])}'")

    # check workers required range
    # valid range is [1, 8]
    if not 1 <= line_info["WorkersRequired"] <= 8:
        log.error(f"line_info has unexpected value '{line_info['WorkersRequired']}'")
        raise ValueError(f"line_info has unexpected value '{line_info['WorkersRequired']}'. "
                         f"Expected value between 1 and 8.")

    # check due date range
    # valid range is [-10, 10]
    if not -10 <= line_info["DueDate"] <= 10:
        log.error(f"line_info has unexpected value '{line_info['DueDate']}'")
        raise ValueError(f"line_info has unexpected value '{line_info['DueDate']}'. "
                         f"Expected value between -10 and 10.")


if __name__ == '__main__':
    import json
    data = None
    # file = resources_dir_path.joinpath("OutputKBv2.json")
    # file = resources_dir_path.joinpath("OutputKB.json")
    file = resources_dir_path.joinpath("OutputKB_Final.json")
    with open(file) as json_file:
        data = json.load(json_file)

    validate_instance(data)
