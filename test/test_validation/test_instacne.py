import pytest
import json
from validation.input_validation import validate_order_info_list_elem, validate_instance
from utils.project_paths import resources_dir_path


def test_valid_worker_preference_list():
    # load OutputKB_Final.json
    path = resources_dir_path.joinpath("OutputKB_Final.json")
    # parse json
    with open(path) as json_file:
        kb_data = json.load(json_file)

    validate_instance(kb_data)

def test_invalid_worker_preference_list():
    # load OutputKB_Final.json
    with pytest.raises(KeyError):
        validate_instance({})
    with pytest.raises(TypeError):
        validate_instance({"OrderInfoList": "123"})
