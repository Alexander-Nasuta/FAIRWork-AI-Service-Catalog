from demonstrator.linear_assignment_solver import allocate_using_linear_assignment_solver
from rest.flaskx_endpoints import _perform_order_to_line_mapping
from utils.project_paths import resources_dir_path


def test_lin_assignment_solver1():
    import json

    data = None
    # file = resources_dir_path.joinpath("OutputKBv2.json")
    # file = resources_dir_path.joinpath("OutputKB.json")
    file = resources_dir_path.joinpath("crf_service_input.json")
    with open(file) as json_file:
        data = json.load(json_file)

    _ = _perform_order_to_line_mapping(data)



def test_lin_assignment_solver2():
    import json

    data = None
    # file = resources_dir_path.joinpath("OutputKBv2.json")
    # file = resources_dir_path.joinpath("OutputKB.json")
    file = resources_dir_path.joinpath("crf_service_input.json")
    with open(file) as json_file:
        data = json.load(json_file)

    data_with_perform_allocation_for_lines = data | {
        "perform_allocation_for_lines": [
            "line 17",
            "line 20",
            "line 24"
        ],
    }

    _ = _perform_order_to_line_mapping(data_with_perform_allocation_for_lines)


def test_lin_assignment_solver3():
    import json

    data = None
    # file = resources_dir_path.joinpath("OutputKBv2.json")
    # file = resources_dir_path.joinpath("OutputKB.json")
    file = resources_dir_path.joinpath("crf_service_input.json")
    with open(file) as json_file:
        data = json.load(json_file)

    data_with_perform_allocation_for_lines = data | {
        "perform_allocation_for_lines": [
            "line 20",
            "line 24"
        ],
    }

    _ = _perform_order_to_line_mapping(data_with_perform_allocation_for_lines)
