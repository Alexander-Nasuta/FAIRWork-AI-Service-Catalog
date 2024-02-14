from copy import deepcopy

import pytest
import json
from validation.input_validation import validate_order_info_list_elem
from utils.project_paths import resources_dir_path


def test_valid_worker_preference_list():
    # load OutputKB_Final.json
    path = resources_dir_path.joinpath("OutputKB_Final.json")
    # parse json
    with open(path) as json_file:
        kb_data = json.load(json_file)

    for order_info_elem in kb_data["OrderInfoList"]:
        validate_order_info_list_elem(order_info_elem)


def test_invalid_worker_preference_list():
    # load OutputKB_Final.json
    path = resources_dir_path.joinpath("OutputKB_Final.json")
    # parse json
    with open(path) as json_file:
        valid_kb_data = json.load(json_file)

    # check add same worker twice
    invalid_kb_data = deepcopy(valid_kb_data)
    for order_info_elem in invalid_kb_data["OrderInfoList"]:
        order_info_elem["WorkerInfoList"][0] = order_info_elem["WorkerInfoList"][1]

        with pytest.raises(ValueError):
            validate_order_info_list_elem(order_info_elem)

    # check for wrong length of worker list
    invalid_kb_data = deepcopy(valid_kb_data)
    for order_info_elem in invalid_kb_data["OrderInfoList"]:
        order_info_elem["WorkerInfoList"].pop()

        with pytest.raises(ValueError):
            validate_order_info_list_elem(order_info_elem)

    # check for missing key
    invalid_kb_data = deepcopy(valid_kb_data)
    for order_info_elem in invalid_kb_data["OrderInfoList"]:
        del order_info_elem["WorkerInfoList"]

        with pytest.raises(KeyError):
            validate_order_info_list_elem(order_info_elem)
    invalid_kb_data = deepcopy(valid_kb_data)
    for order_info_elem in invalid_kb_data["OrderInfoList"]:
        del order_info_elem["WorkerInfoList"]

        with pytest.raises(KeyError):
            validate_order_info_list_elem(order_info_elem)

    # check for wrong type
    for order_info_elem in invalid_kb_data["OrderInfoList"]:
        order_info_elem["WorkerInfoList"] = 123

        with pytest.raises(TypeError):
            validate_order_info_list_elem(order_info_elem)
    invalid_kb_data = deepcopy(valid_kb_data)
    for order_info_elem in invalid_kb_data["OrderInfoList"]:
        order_info_elem["WorkerInfoList"] = 321

        with pytest.raises(TypeError):
            validate_order_info_list_elem(order_info_elem)


