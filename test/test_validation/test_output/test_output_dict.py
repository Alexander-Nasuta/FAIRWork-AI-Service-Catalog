import json

import pytest

from utils.project_paths import resources_dir_path
from validation.output_validation import validate_output_dict


# Test for validate_dict function
@pytest.mark.parametrize(
    "dict_input, expected",
    [
        ({
            "AllocationList": [
                {
                    "LineId": "17",
                    "WorkersRequired": 6,
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
                            "Id": "100045",
                            "Availability": "True",
                            "MedicalCondition": "True",
                            "UTEExperience": "False",
                            "WorkerResilience": 0.66,
                            "WorkerPreference": 0.83
                        },
                        {
                            "Id": "100070",
                            "Availability": "True",
                            "MedicalCondition": "True",
                            "UTEExperience": "False",
                            "WorkerResilience": 0.7,
                            "WorkerPreference": 0.82
                        },
                        {
                            "Id": "100011",
                            "Availability": "True",
                            "MedicalCondition": "True",
                            "UTEExperience": "True",
                            "WorkerResilience": 0.7,
                            "WorkerPreference": 0.8
                        },
                        {
                            "Id": "100067",
                            "Availability": "True",
                            "MedicalCondition": "True",
                            "UTEExperience": "False",
                            "WorkerResilience": 0.5,
                            "WorkerPreference": 0.8
                        },
                        {
                            "Id": "100148",
                            "Availability": "True",
                            "MedicalCondition": "True",
                            "UTEExperience": "True",
                            "WorkerResilience": 0.8,
                            "WorkerPreference": 0.5
                        }
                    ]
                },
            ]
        }, None),
        ({
            "AllocationList": "Not a list"  # should be a list
        }, TypeError),
        ({
            # "AllocationList": []  # missing key
        }, KeyError),
        ({
             "AllocationList": [
                 {
                     "LineId": "17",
                     "WorkersRequired": 6,
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
                             "Id": "100060", # duplicate id
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "False",
                             "WorkerResilience": 0.66,
                             "WorkerPreference": 0.83
                         },
                         {
                             "Id": "100070",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "False",
                             "WorkerResilience": 0.7,
                             "WorkerPreference": 0.82
                         },
                         {
                             "Id": "100011",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "True",
                             "WorkerResilience": 0.7,
                             "WorkerPreference": 0.8
                         },
                         {
                             "Id": "100067",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "False",
                             "WorkerResilience": 0.5,
                             "WorkerPreference": 0.8
                         },
                         {
                             "Id": "100148",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "True",
                             "WorkerResilience": 0.8,
                             "WorkerPreference": 0.5
                         }
                     ]
                 },
                 {
                     "LineId": "18",
                     "WorkersRequired": 6,
                     "Workers": [
                         {
                             "Id": "100054",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "False",
                             "WorkerResilience": 0.8,
                             "WorkerPreference": 0.9
                         },
                         {
                             "Id": "100112",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "False",
                             "WorkerResilience": 0.3,
                             "WorkerPreference": 0.9
                         },
                         {
                             "Id": "100141",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "True",
                             "WorkerResilience": 0.68,
                             "WorkerPreference": 0.8
                         },
                         {
                             "Id": "100142",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "True",
                             "WorkerResilience": 0.7,
                             "WorkerPreference": 0.8
                         },
                         {
                             "Id": "100147",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "False",
                             "WorkerResilience": 0.6,
                             "WorkerPreference": 0.8
                         },
                         {
                             "Id": "100063",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "False",
                             "WorkerResilience": 0.8,
                             "WorkerPreference": 0.55
                         }
                     ]
                 },
                 {
                     "LineId": "20",
                     "WorkersRequired": 4,
                     "Workers": [
                         {
                             "Id": "100111",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "False",
                             "WorkerResilience": 0.5,
                             "WorkerPreference": 1
                         },
                         {
                             "Id": "100022",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "False",
                             "WorkerResilience": 0.9,
                             "WorkerPreference": 0.9
                         },
                         {
                             "Id": "100023",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "False",
                             "WorkerResilience": 0.3,
                             "WorkerPreference": 0.9
                         },
                         {
                             "Id": "100029",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "False",
                             "WorkerResilience": 0.8,
                             "WorkerPreference": 0.8
                         }
                     ]
                 }
             ]
         }, ValueError),
        ({
             "AllocationList": [
                 {
                     "LineId": "17",
                     "WorkersRequired": 2, # not matching with the number of workers
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
                             "Id": "100060", # duplicate id
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "False",
                             "WorkerResilience": 0.66,
                             "WorkerPreference": 0.83
                         },
                         {
                             "Id": "100070",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "False",
                             "WorkerResilience": 0.7,
                             "WorkerPreference": 0.82
                         },
                         {
                             "Id": "100011",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "True",
                             "WorkerResilience": 0.7,
                             "WorkerPreference": 0.8
                         },
                         {
                             "Id": "100067",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "False",
                             "WorkerResilience": 0.5,
                             "WorkerPreference": 0.8
                         },
                         {
                             "Id": "100148",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "True",
                             "WorkerResilience": 0.8,
                             "WorkerPreference": 0.5
                         }
                     ]
                 },
                 {
                     "LineId": "18",
                     "WorkersRequired": 6,
                     "Workers": [
                         {
                             "Id": "100054",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "False",
                             "WorkerResilience": 0.8,
                             "WorkerPreference": 0.9
                         },
                         {
                             "Id": "100112",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "False",
                             "WorkerResilience": 0.3,
                             "WorkerPreference": 0.9
                         },
                         {
                             "Id": "100141",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "True",
                             "WorkerResilience": 0.68,
                             "WorkerPreference": 0.8
                         },
                         {
                             "Id": "100142",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "True",
                             "WorkerResilience": 0.7,
                             "WorkerPreference": 0.8
                         },
                         {
                             "Id": "100147",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "False",
                             "WorkerResilience": 0.6,
                             "WorkerPreference": 0.8
                         },
                         {
                             "Id": "100063",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "False",
                             "WorkerResilience": 0.8,
                             "WorkerPreference": 0.55
                         }
                     ]
                 },
                 {
                     "LineId": "20",
                     "WorkersRequired": 4,
                     "Workers": [
                         {
                             "Id": "100111",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "False",
                             "WorkerResilience": 0.5,
                             "WorkerPreference": 1
                         },
                         {
                             "Id": "100022",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "False",
                             "WorkerResilience": 0.9,
                             "WorkerPreference": 0.9
                         },
                         {
                             "Id": "100023",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "False",
                             "WorkerResilience": 0.3,
                             "WorkerPreference": 0.9
                         },
                         {
                             "Id": "100029",
                             "Availability": "True",
                             "MedicalCondition": "True",
                             "UTEExperience": "False",
                             "WorkerResilience": 0.8,
                             "WorkerPreference": 0.8
                         }
                     ]
                 }
             ]
         }, ValueError),
    ],
)
def test_validate_dict(dict_input, expected):
    if expected is None:
        assert validate_output_dict(dict_input) is None
    else:
        with pytest.raises(expected):
            validate_output_dict(dict_input)


def test_parsed_instance():
    # load OutputKB_Final.json
    path = resources_dir_path.joinpath("ServiceOutput.json")
    # parse json
    with open(path) as json_file:
        data = json.load(json_file)

    validate_output_dict(data)