import pytest
from validation.input_validation import validate_worker_preference_list

# validate_worker_preference_list is for a single worker
@pytest.mark.parametrize(
    "preference_list, expected",
    [
        ([
             {"LineId": "17", "Value": 0.1},
             {"LineId": "18", "Value": 0.0},
             {"LineId": "20", "Value": 0.9}
         ], None),
        ([
             {"LineId": "17", "Value": 1/3},
             {"LineId": "18", "Value": 1/3},
             {"LineId": "20", "Value": 1/3}
         ], None),
        ([
             {"LineId": "18", "Value": 0},
             {"LineId": "17", "Value": 1},
             {"LineId": "20", "Value": 0}
         ], None),
        ([
             {"LineId": "17", "Value": 1},
         ], None),
        ([
             {"LineId": "17", "Value": 1},
             {"LineId": "20", "Value": 0},
         ], None),
    ],
)
def test_valid_worker_preference_list(preference_list, expected):
    res = validate_worker_preference_list(preference_list)
    assert res is None

@pytest.mark.parametrize(
    "preference_list, expected",
    [
        ([
             {"LineId": "17", "Value": 0},
             {"LineId": "18", "Value": 0.0},
             {"LineId": "20", "Value": 0}
         ], ValueError),
        ([
             {"LineId": "17", "Value": 0.1},
             {"LineId": "18", "Value": 0.9},
             {"LineId": "20", "Value": 0.00001}
         ], ValueError),
        ([
             {"LineId": 123, "Value": 0},
             {"LineId": "17", "Value": 1},
             {"LineId": "20", "Value": 0}
         ], TypeError),
        ([
             {"LineId": True, "Value": 0},
             {"LineId": "17", "Value": 1},
             {"LineId": "20", "Value": 0}
         ], TypeError),
        ([
             {"Value": 0},
             {"LineId": "17", "Value": 1},
             {"LineId": "20", "Value": 0}
         ], KeyError),
        ([
             {"LineId": 17, "Value": 0.1},
             {"LineId": "18", "Value": 0.0},
             {"LineId": "20", "Value": 0.9}
         ], TypeError),
    ],
)
def test_invalid_worker_preference_list(preference_list, expected):
    with pytest.raises(expected):
        validate_worker_preference_list(preference_list)


