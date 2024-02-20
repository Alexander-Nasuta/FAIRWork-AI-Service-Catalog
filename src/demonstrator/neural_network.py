import pprint
import uuid

import torch

import numpy as np
import pathlib as pl
import pandas as pd
import wandb

from torch import nn, optim
from torch.utils.data import Dataset

from utils.logger import log
from utils.project_paths import historic_data_dir_path, trained_models_dir_path, resources_dir_path
from validation.input_validation import validate_instance
from validation.output_validation import validate_output_dict


class DemonstratorNeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer_1 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer_1(x))
        x = self.output_layer(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, data_directory: str | pl.Path):
        self.data_directory = pl.Path(data_directory)
        self.filenames = list(self.data_directory.glob("*.csv"))

        if not len(self.filenames):
            raise ValueError(f"no .csv files found in '{self.data_directory}'")

        sample_file = pd.read_csv(self.filenames[0])
        # drop index column
        sample_file = sample_file.drop(sample_file.columns[0], axis=1)
        sample_y = sample_file.pop("FinalAllocation")

        self.n_y_params, *_ = np.ravel(sample_y.to_numpy()).shape
        self.n_x_params, *_ = np.ravel(sample_file.to_numpy()).shape

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):  # idx means index of the chunk.
        # In this method, we do all the preprocessing.
        # First read data from files in a chunk. Preprocess it. Extract labels. Then return data and labels.
        file = self.filenames[idx]

        df = pd.read_csv(open(file, 'r'))
        # drop index column
        df = df.drop(df.columns[0], axis=1)
        y_data = np.ravel(df.pop("FinalAllocation").to_numpy())
        x_data = np.ravel(df.to_numpy())

        # The following condition is actually needed in Pytorch. Otherwise, for our particular example, the iterator will be an infinite loop.
        # Readers can verify this by removing this condition.
        if idx == self.__len__():
            raise IndexError

        return x_data, y_data


def train_demonstrator_model(n_epochs: int) -> None:
    model_name = f"model_nn_{uuid.uuid4()}"
    # list all .csv files in the training-data directory
    data_dir_path = historic_data_dir_path
    dataset = CustomDataset(data_directory=data_dir_path)

    # Define configuration
    config = {
        "learning_rate": 0.001,
        "epochs": n_epochs,
        "hidden_dim": 100,
        "batch_size": 5,
        "optimizer": "adam",
        "loss_function": "MSE",
        "model_name": model_name,
        "dataset.n_x_params": dataset.n_x_params,
        "dataset.n_y_params": dataset.n_y_params,
    }
    with wandb.init(project='test', config=config) as run:
        model = DemonstratorNeuralNet(
            input_dim=dataset.n_x_params,
            hidden_dim=config["hidden_dim"],
            output_dim=dataset.n_y_params
        )
        log.info(f"model: {model}")

        # define pytorch adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

        # define pytorch mean squared error loss function
        loss_fn = nn.MSELoss()

        # define pytorch data loader
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

        wandb.watch(model)

        # define pytorch training loop
        optimizer_step = 0
        for epoch in range(n_epochs):
            for batch_idx, (x, y) in enumerate(data_loader):
                # x = x.float()
                # y = y.float()
                optimizer.zero_grad()
                y_pred = model(x.to(torch.float32)).to(torch.float32)
                loss = loss_fn(y_pred, y.to(torch.float32))
                loss.backward()
                optimizer.step()
                log.info(f"epoch: {epoch}, batch_idx: {batch_idx}, loss: {loss.item()}")
                wandb.log({
                    "loss": loss.item(),
                    "epoch": epoch,
                    "optimizer_step": optimizer_step
                })
                optimizer_step += 1

        # save model
        model_path = trained_models_dir_path.joinpath(f"{model_name}.pt")
        torch.save(model.state_dict(), model_path)


def get_model(model_name: str = "model_nn_8b80629d-83ca-420d-9b63-c6b820bcc27d",
              hidden_dim: int = 100) -> DemonstratorNeuralNet:
    model_path = trained_models_dir_path.joinpath(f"{model_name}.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"could not find model '{model_name}'.")
    # load model

    x_dim = 1272
    y_dim = 159

    model = DemonstratorNeuralNet(
        input_dim=x_dim,
        hidden_dim=hidden_dim,
        output_dim=y_dim
    )
    model.load_state_dict(torch.load(model_path))

    return model


def get_solution(instance: dict):
    model = get_model()

    # covert instance to pandas dataframe
    intermediate_res = {}  # dict of dataframes
    already_allocated_workers = set()
    for order_info in instance["OrderInfoList"]:
        line_info = order_info["LineInfo"]
        current_line = line_info["LineId"]
        worker_info_list = order_info["WorkerInfoList"]
        data = []
        for worker_info in worker_info_list:
            # ID,Availability,MedicalCondition,UTEExperience,WorkerResilience,WorkerPreference,ProductionPriority,DueDate,FinalAllocation
            data.append({
                "Id": worker_info["Id"],
                "Availability": worker_info["Availability"] == "True",
                "MedicalCondition": worker_info["MedicalCondition"] == "True",
                "UTEExperience": worker_info["UTEExperience"] == "True",
                "WorkerResilience": worker_info["WorkerResilience"],
                "ProductionPriority": line_info["ProductionPriority"] == "True",
                "DueDate": line_info["DueDate"],
                "WorkerPreference": sum([
                    e["Value"]
                    for e in worker_info["WorkerPreference"]
                    if e["LineId"] == line_info["LineId"]
                ])
            })
        df = pd.DataFrame(data).astype(float)
        # predict using model
        x = np.ravel(df.to_numpy())
        y_pred = model(torch.tensor(x, dtype=torch.float32)).to(torch.float32)
        # add y_pred to df as a new column 'FinalAllocation'
        df["FinalAllocation"] = y_pred.detach().numpy()

        # set coloumn id to int
        df["Id"] = df["Id"].astype(int)

        n_workers = line_info["WorkersRequired"]

        available_workers_df = df[df["Availability"] == 1]
        # only keep workers with medical condition == True
        available_workers_df = available_workers_df[available_workers_df["MedicalCondition"] == 1]

        # drop rows where worker is already allocated
        available_workers_df = available_workers_df[~available_workers_df["Id"].isin(already_allocated_workers)]

        # sort by FinalAllocation and keep #n_workers elements
        available_workers_df = available_workers_df.nlargest(n_workers, "FinalAllocation")

        # add allocated workers to already_allocated_workers
        already_allocated_workers.update(available_workers_df["Id"])

        # set cols ID, Availability MedicalCondition UTEExperience, ProductionPriority to str
        # 1.0 -> 'True', 0.0 -> 'False'
        available_workers_df["Availability"] = available_workers_df["Availability"].astype(bool).astype(str)
        available_workers_df["MedicalCondition"] = available_workers_df["MedicalCondition"].astype(bool).astype(str)
        available_workers_df["UTEExperience"] = available_workers_df["UTEExperience"].astype(bool).astype(str)
        available_workers_df["ProductionPriority"] = available_workers_df["ProductionPriority"].astype(bool).astype(str)
        # set DueDate to int
        available_workers_df["DueDate"] = available_workers_df["DueDate"].astype(int)
        # set ID to str
        available_workers_df["Id"] = available_workers_df["Id"].astype(str)

        # drop FinalAllocation column since it shall not be present in the output
        available_workers_df = available_workers_df.drop("FinalAllocation", axis=1)

        intermediate_res[current_line] = available_workers_df

    res = []
    for order_info_elem in instance["OrderInfoList"]:
        line_info = order_info_elem["LineInfo"]
        line_id = line_info["LineId"]
        res.append({
            "LineId": line_info["LineId"],
            "WorkersRequired": line_info["WorkersRequired"],
            "Workers": intermediate_res[line_id].to_dict("records")
        })
    res = {"AllocationList": res}

    log.info(f'Suggested solution by NN: \n{pprint.pformat(res)}', extra=res)
    return res


if __name__ == '__main__':

    import json

    data: dict
    file = resources_dir_path.joinpath("OutputKB_Final.json")
    with open(file) as json_file:
        data = json.load(json_file)

    validate_instance(data)

    res = get_solution(instance=data)

    validate_output_dict(res)
