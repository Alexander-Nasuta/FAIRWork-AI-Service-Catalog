import pathlib as pl
import pandas as pd
import uuid

from data_generator.generate_instance import generate_instance
from demonstrator.linear_assignment_solver import allocate_using_linear_assignment_solver
from utils.project_paths import historic_data_dir_path
from utils.logger import log
from validation.input_validation import validate_instance
from validation.output_validation import validate_output_dict


def generate_single_line_simulated_historic_data_using_lin_solver(n_files=2) -> None:
    pl.Path(historic_data_dir_path).mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        file_name = f"{uuid.uuid4()}.csv"
        log.info(f"[{i + 1}/{n_files}] generating simulated data...")

        random_instance = generate_instance()

        # validate instance
        validate_instance(random_instance)

        # solve instance via linear assignment solver
        solution = allocate_using_linear_assignment_solver(random_instance)
        # extract allocated workers
        allocated_workers_ids = set()
        for allocation in solution["AllocationList"]:
            for worker in allocation["Workers"]:
                if worker["Id"] in allocated_workers_ids:
                    raise ValueError(f"Duplicate ID found: {worker['Id']}")
                allocated_workers_ids.add(worker["Id"])

        # validate solution
        validate_output_dict(solution)

        # construct csv data using pandas
        data = []
        line_info = random_instance["OrderInfoList"][0]["LineInfo"]
        current_line = line_info["LineId"]
        worker_info_list = random_instance["OrderInfoList"][0]["WorkerInfoList"]
        for worker_info in worker_info_list:
            data.append({
                "Id": worker_info["Id"],
                "Availability": worker_info["Availability"] == "True",
                "MedicalCondition": worker_info["MedicalCondition"] == "True",
                "UTEExperience": worker_info["UTEExperience"] == "True",
                "WorkerResilience": worker_info["WorkerResilience"],
                "WorkerPreference": sum([
                    e["Value"]
                    for e in worker_info["WorkerPreference"]
                    if e["LineId"] == current_line
                ]),
                "ProductionPriority": line_info["ProductionPriority"] == "True",
                "DueDate": line_info["DueDate"],
                "FinalAllocation": int(worker_info["Id"] in allocated_workers_ids)
            })
        df = pd.DataFrame(data).astype(float)
        path = pl.Path(historic_data_dir_path).joinpath(file_name)

        log.info(f"saving df as file '{file_name}'")
        # print(df.head())
        df.to_csv(path_or_buf=path)


if __name__ == '__main__':
    generate_single_line_simulated_historic_data_using_lin_solver(n_files=100)
