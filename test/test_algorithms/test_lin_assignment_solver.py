from demonstrator.linear_assignment_solver import allocate_using_linear_assignment_solver
from utils.project_paths import resources_dir_path


def test_lin_assignment_solver():
    import json

    data = None
    # file = resources_dir_path.joinpath("OutputKBv2.json")
    # file = resources_dir_path.joinpath("OutputKB.json")
    file = resources_dir_path.joinpath("OutputKB_Final.json")
    with open(file) as json_file:
        data = json.load(json_file)

    res = allocate_using_linear_assignment_solver(data)
