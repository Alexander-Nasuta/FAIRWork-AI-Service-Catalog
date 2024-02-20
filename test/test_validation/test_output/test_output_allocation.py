import pytest

from validation.output_validation import validate_allocation


# Test for validate_allocation function
@pytest.mark.parametrize(
    "allocation, expected",
    [
        ({
             "LineId": "Line1",
             "WorkersRequired": 2,
             "Workers": [
                 {
                     "Id": "100060",
                     "Availability": "True",
                     "MedicalCondition": "True",
                     "UTEExperience": "False",
                     "WorkerResilience": 0.87,
                     "WorkerPreference": 0.9
                 },
                 {
                     "Id": "100061",
                     "Availability": "True",
                     "MedicalCondition": "True",
                     "UTEExperience": "False",
                     "WorkerResilience": 0.87,
                     "WorkerPreference": 0.9
                 }
             ]
         }, None),
        ({
             "LineId": "Line1",
             "WorkersRequired": 2,
             # "Workers": []  # missing key
         }, KeyError),
        ({
             "LineId": "Line1",
             "WorkersRequired": 2,
             "Workers": "Not a list"  # should be a list
         }, TypeError),
    ],
)
def test_validate_allocation(allocation, expected):
    if expected is None:
        assert validate_allocation(allocation) is None
    else:
        with pytest.raises(expected):
            validate_allocation(allocation)