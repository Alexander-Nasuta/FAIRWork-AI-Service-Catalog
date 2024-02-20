from demonstrator.neural_network import get_solution
from utils.project_paths import resources_dir_path
from validation.output_validation import validate_output_dict


def test_lin_assignment_solver():
    import json

    data = None
    # file = resources_dir_path.joinpath("OutputKBv2.json")
    # file = resources_dir_path.joinpath("OutputKB.json")
    file = resources_dir_path.joinpath("OutputKB_Final.json")
    with open(file) as json_file:
        data = json.load(json_file)

    res = get_solution(data)
    validate_output_dict(res)
