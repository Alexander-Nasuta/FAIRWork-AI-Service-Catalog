import pprint
from utils.logger import log

from numbers import Real

from utils.project_paths import resources_dir_path


def validate_worker(worker):
    if not isinstance(worker, dict):
        log.error("Worker must be a dictionary", extra=worker)
        raise TypeError("Worker must be a dictionary")
    if "Id" not in worker:
        log.error("Worker must have an 'Id'", extra=worker)
        raise KeyError("Worker must have an 'Id'")
    if not isinstance(worker["Id"], str):
        log.error("'Id' must be a string", extra=worker)
        raise TypeError("'Id' must be a string")
    if "Availability" not in worker:
        log.error("Worker must have 'Availability'", extra=worker)
        raise KeyError("Worker must have 'Availability'")
    if not isinstance(worker["Availability"], str):
        log.error("'Availability' must be a string", extra=worker)
        raise TypeError("'Availability' must be a string")
    if "MedicalCondition" not in worker:
        log.error("Worker must have 'MedicalCondition'", extra=worker)
        raise KeyError("Worker must have 'MedicalCondition'")
    if not isinstance(worker["MedicalCondition"], str):
        log.error("'MedicalCondition' must be a string", extra=worker)
        raise TypeError("'MedicalCondition' must be a string")
    if "UTEExperience" not in worker:
        log.error("Worker must have 'UTEExperience'", extra=worker)
        raise KeyError("Worker must have 'UTEExperience'")
    if not isinstance(worker["UTEExperience"], str):
        log.error("'UTEExperience' must be a string", extra=worker)
        raise TypeError("'UTEExperience' must be a string")
    if "WorkerResilience" not in worker:
        log.error("Worker must have 'WorkerResilience'", extra=worker)
        raise KeyError("Worker must have 'WorkerResilience'")
    if not isinstance(worker["WorkerResilience"], Real):
        log.error("'WorkerResilience' must be a Real", extra=worker)
        raise TypeError("'WorkerResilience' must be a Real")
    if "WorkerPreference" not in worker:
        log.error("Worker must have 'WorkerPreference'", extra=worker)
        raise KeyError("Worker must have 'WorkerPreference'")
    if not isinstance(worker["WorkerPreference"], Real):
        log.error("'WorkerPreference' must be a Real", extra=worker)
        raise TypeError("'WorkerPreference' must be a Real")

    if worker["Availability"] != 'True':
        log.error("'Availability' must be 'True'", extra=worker)
        raise ValueError("'Availability' must be 'True'")

    if worker["MedicalCondition"] != 'True':
        log.warning(f"Worker with Id {worker['Id']} is present in an allocation, "
                    f"but It's 'MedicalCondition' is not 'True'", extra=worker)


def validate_allocation(allocation):
    if not isinstance(allocation, dict):
        log.error("Allocation must be a dictionary", extra=allocation)
        raise TypeError("Allocation must be a dictionary")
    if "LineId" not in allocation:
        log.error("Allocation must have a 'LineId'", extra=allocation)
        raise KeyError("Allocation must have a 'LineId'")
    if not isinstance(allocation["LineId"], str):
        log.error("'LineId' must be a string", extra=allocation)
        raise TypeError("'LineId' must be a string")
    if "WorkersRequired" not in allocation:
        log.error("Allocation must have 'WorkersRequired'", extra=allocation)
        raise KeyError("Allocation must have 'WorkersRequired'")
    if not isinstance(allocation["WorkersRequired"], int):
        log.error("'WorkersRequired' must be an integer", extra=allocation)
        raise TypeError("'WorkersRequired' must be an integer")
    if "Workers" not in allocation:
        log.error("Allocation must have 'Workers'", extra=allocation)
        raise KeyError("Allocation must have 'Workers'")
    if not isinstance(allocation["Workers"], list):
        log.error("'Workers' must be a list", extra=allocation)
        raise TypeError("'Workers' must be a list")

    if len(allocation["Workers"]) != allocation["WorkersRequired"]:
        log.error("Number of workers in 'Workers' list is not equal to 'WorkersRequired'", extra=allocation)
        raise ValueError(f"Number of workers in 'Workers' list is not equal to 'WorkersRequired'")

    for worker in allocation["Workers"]:
        validate_worker(worker)


def validate_output_dict(data):
    if not isinstance(data, dict):
        raise TypeError("Data must be a dictionary")
    if "AllocationList" not in data:
        raise KeyError("Data must have 'AllocationList'")
    if not isinstance(data["AllocationList"], list):
        raise TypeError("'AllocationList' must be a list")

    validate_unique_ids(data)

    for allocation in data["AllocationList"]:
        validate_allocation(allocation)


def validate_unique_ids(data):
    id_set = set()
    for allocation in data["AllocationList"]:
        for worker in allocation["Workers"]:
            if worker["Id"] in id_set:
                raise ValueError(f"Duplicate ID found: {worker['Id']}")
            id_set.add(worker["Id"])


if __name__ == '__main__':
    import json

    data = None
    file = resources_dir_path.joinpath("ServiceOutput.json")
    with open(file) as json_file:
        data = json.load(json_file)

    print(pprint.pformat(data))
    validate_output_dict(data)
