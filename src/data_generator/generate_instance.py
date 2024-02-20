import pprint
import random

from validation.input_validation import validate_instance


def generate_worker_preference() -> list[dict]:
    line_ids = ["17", "18", "20"]
    random_values = [random.random() for _ in line_ids]

    # Normalize the values so they add up to 1.0
    total = sum(random_values)
    normalized_values = [value / total for value in random_values]

    # Create the list of dictionaries
    worker_preference = [{"LineId": line_id, "Value": value} for line_id, value in zip(line_ids, normalized_values)]

    return worker_preference


def generate_worker(id: str | int) -> dict:
    worker = {
        "Id": f"{id}",
        "Availability": str(bool(random.getrandbits(1))),
        "MedicalCondition": str(bool(random.getrandbits(1))),
        "UTEExperience": str(bool(random.getrandbits(1))),
        "WorkerResilience": round(random.uniform(0, 1), 2),
        "WorkerPreference": generate_worker_preference()
    }

    return worker


def generate_line_info() -> dict:
    line_info = {
        "LineId": random.choice(["17", "18", "20"]),
        "Geometry": str(random.randint(1000000000, 9999999999)),
        "ProductionPriority": str(bool(random.getrandbits(1))),
        "DueDate": random.randint(1, 10),
        "WorkersRequired": random.randint(1, 8)
    }

    return line_info


def generate_order_info_list_elem() -> dict:
    worker_info_list = [generate_worker(id) for id in range(100001, 100160)]
    line_info = generate_line_info()

    result = {
        "WorkerInfoList": worker_info_list,
        "LineInfo": line_info
    }

    return result


def generate_instance() -> dict:
    random_line_info = generate_order_info_list_elem()
    return {"OrderInfoList": [random_line_info]}


if __name__ == '__main__':
    res = generate_instance()
    validate_instance(res)
