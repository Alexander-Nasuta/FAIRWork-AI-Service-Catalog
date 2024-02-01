import pytest
from validation.input_validation import validate_line_info

@pytest.mark.parametrize(
    "line_info, expected",
    [
        ({
             "LineId": "17",
             "Geometry": "1343314080",
             "ProductionPriority": "False",
             "DueDate": 3,
             "WorkersRequired": 6
         }, None),
        ({
             "LineId": "17",
             "Geometry": "1343314080",
             "ProductionPriority": "True",
             "DueDate": 3,
             "WorkersRequired": 6
         }, None),
        ({
             "LineId": "17",
             "Geometry": "1343314080",
             "ProductionPriority": "True",
             "DueDate": 10,
             "WorkersRequired": 6
         }, None),
        ({
             "LineId": "17",
             "Geometry": "1343314080",
             "ProductionPriority": "True",
             "DueDate": -10,
             "WorkersRequired": 6
         }, None),
        ({
             "LineId": "17",
             "Geometry": "1343314080",
             "ProductionPriority": "True", # valid values are "True" or "False"
             "DueDate": 3, # valid range is [-10, 10] (both inclusive)
             "WorkersRequired": 8 # valid range is [1, 8]
        }, None),
        ({
             "LineId": "17",
             "Geometry": "1343314080",
             "ProductionPriority": "True", # valid values are "True" or "False"
             "DueDate": 3, # valid range is [-10, 10] (both inclusive)
             "WorkersRequired": 1 # valid range is [1, 8] (both inclusive)
        }, None)
    ],
)
def test_valid_line_info(line_info, expected):

    res = validate_line_info(line_info)
    assert res is None

@pytest.mark.parametrize(
    "line_info, expected",
    [
        ({
             "LineId": "17",
             "Geometry": "1343314080",
             "ProductionPriority": False, # should be string
             "DueDate": 3,
             "WorkersRequired": 8
         }, TypeError),
        ({
             "LineId": "17",
             "Geometry": "1343314080",
             "ProductionPriority": 9000, # should be string
             "DueDate": 3,
             "WorkersRequired": 8
         }, TypeError),
        ({
             "LineId": "17",
             "Geometry": "1343314080",
             "ProductionPriority": "True",
             "DueDate": -11,  # should be in range [-10,10] (both inclusive)
             "WorkersRequired": 8
         }, ValueError),
        ({
             "LineId": "17",
             "Geometry": "1343314080",
             "ProductionPriority": "True",
             "DueDate": -11,  # should be in range [-10,10] (both inclusive)
             "WorkersRequired": 8
         }, ValueError),
        ({
             "LineId": "17",
             "Geometry": "1343314080",
             "ProductionPriority": "True", # valid values are "True" or "False"
             "DueDate": 3, # valid range is [-10, 10] (both inclusive)
             "WorkersRequired": 9 # valid range is [1, 8] (both inclusive)
        }, ValueError),
        ({
             "LineId": "17",
             "Geometry": "1343314080",
             "ProductionPriority": "True", # valid values are "True" or "False"
             "DueDate": 3, # valid range is [-10, 10] (both inclusive)
             "WorkersRequired": 0 # valid range is [1, 8] (both inclusive)
        }, ValueError)
    ],
)
def test_validate_line_info_with_invalid_values(line_info, expected):
    with pytest.raises(expected):
        validate_line_info(line_info)
