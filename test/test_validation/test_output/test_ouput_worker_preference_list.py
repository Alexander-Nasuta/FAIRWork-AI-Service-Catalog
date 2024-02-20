import pytest

from validation.output_valiation import validate_worker


# Test for validate_worker function
@pytest.mark.parametrize(
    "worker, expected",
    [
        ({
             "Id": "100060",
             "Availability": "True",
             "MedicalCondition": "True",
             "UTEExperience": "False",
             "WorkerResilience": 0.87,
             "WorkerPreference": 0.9
         }, None),
        ({
             "Id": "100060",
             "Availability": "True",
             "MedicalCondition": "True",
             "UTEExperience": "False",
             "WorkerResilience": "0.87",  # should be float
             "WorkerPreference": 0.9
         }, TypeError),
        ({
             "Id": "100060",
             "Availability": "True",
             "MedicalCondition": "True",
             "UTEExperience": "False",
             "WorkerResilience": 0.87,
             # "WorkerPreference": 0.9  # missing key
         }, KeyError),
    ],
)
def test_validate_worker(worker, expected):
    if expected is None:
        assert validate_worker(worker) is None
    else:
        with pytest.raises(expected):
            validate_worker(worker)