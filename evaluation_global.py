import warnings
import os
from model_function import *
from model_utils import *
from utils import *
from torch.utils.data import DataLoader
import torch.nn.functional as Fin
import timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# from torch_geometric.data import Data
from torchdiffeq import odeint as odeint
import matplotlib

matplotlib.use("Agg")
import argparse
import sys
import time
import torch
import torch.optim as optim
import random
from tqdm.cli import tqdm

SOLVERS = [
    "dopri8",
    "dopri5",
    "bdf",
    "rk4",
    "midpoint",
    "adams",
    "explicit_adams",
    "fixed_adams",
    "adaptive_heun",
]
parser = argparse.ArgumentParser("ClimODE")

parser.add_argument("--solver", type=str, default="euler", choices=SOLVERS)
parser.add_argument("--atol", type=float, default=5e-3)
parser.add_argument("--rtol", type=float, default=5e-3)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument(
    "--step_size", type=float, default=None, help="Optional fixed step size."
)
parser.add_argument("--scale", type=int, default=0)
parser.add_argument("--days", type=int, default=3)
parser.add_argument("--spectral", type=int, default=0, choices=[0, 1])
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cwd = os.getcwd()
train_time_scale = slice("2006", "2016")
val_time_scale = slice("2016", "2016")
test_time_scale = slice("2017", "2018")

paths_to_data = [
    str(cwd) + "/era5_data/geopotential_500/*.nc",
    str(cwd) + "/era5_data/temperature_850/*.nc",
    str(cwd) + "/era5_data/2m_temperature/*.nc",
    str(cwd) + "/era5_data/10m_u_component_of_wind/*.nc",
    str(cwd) + "/era5_data/10m_v_component_of_wind/*.nc",
    str(cwd) + "/era5_data/v_component_of_wind/*.nc",
    str(cwd) + "/era5_data/u_component_of_wind/*.nc",
    str(cwd) + "/era5_data/relative_humidity/*.nc",
    str(cwd) + "/era5_data/specific_humidity/*.nc",
]
const_info_path = [str(cwd) + "/era5_data/constants/constants_5.625deg.nc"]
levels = ["z", "t", "t2m", "u10", "v10"]
paths_to_data = paths_to_data[0:5]
levels = levels[0:5]
assert len(paths_to_data) == len(
    levels
), "Paths to different type of data must be same as number of types of observations"
print("############################ Data is loading ###########################")
Final_train_data = 0
Final_val_data = 0
Final_test_data = 0
max_lev = []
min_lev = []
for idx, data in enumerate(paths_to_data):
    Train_data, Val_data, Test_data, time_steps, lat, lon, mean, std, time_stamp = (
        get_train_test_data_without_scales_batched(
            data,
            train_time_scale,
            val_time_scale,
            test_time_scale,
            levels[idx],
            args.spectral,
        )
    )
    max_lev.append(mean)
    min_lev.append(std)
    if idx == 0:
        Final_train_data = Train_data
        Final_val_data = Val_data
        Final_test_data = Test_data
    else:
        Final_train_data = torch.cat([Final_train_data, Train_data], dim=2)
        Final_val_data = torch.cat([Final_val_data, Val_data], dim=2)
        Final_test_data = torch.cat([Final_test_data, Test_data], dim=2)

print("Length of training data", len(Final_train_data))
print("Length of validation data", len(Final_val_data))
print("Length of testing data", len(Final_test_data))
const_channels_info, lat_map, lon_map = add_constant_info(const_info_path)

if args.spectral == 1:
    print("############## Running the Model in Spectral Domain ####################")
H, W = Train_data.shape[3], Train_data.shape[4]
clim = torch.mean(Final_test_data, dim=0)
Test_loader = DataLoader(Final_test_data[2:], batch_size=8, shuffle=False)
time_loader = DataLoader(time_steps[2:], batch_size=8, shuffle=False)
time_idx_steps = torch.tensor([i for i in range(365 * 4)]).view(-1, 1)
time_idx = DataLoader(time_idx_steps[2:], batch_size=8, shuffle=False, pin_memory=False)

total_time_len = len(time_steps[2:])
total_time_steps = time_steps[2:].numpy().flatten().tolist()
num_years = 2
Final_train_data = 0
Final_val_data = 0

# vel_test= torch.from_numpy(np.load('### Test velocity here'))
vel_train, vel_val, vel_test = load_velocity(
    ["train_10year_2day_mm", "val_10year_2day_mm", "test_10year_2day_mm"]
)
# NOTE: Absolute garbage - code didn't have a proper path here, mising a slash...
model = torch.load(
    str(cwd) + "/checkpoints/ClimODE_global.pt", map_location=torch.device("cpu")
).to(device)
print(model)

org_time = 1
RMSD = []
RMSD_lat_lon = []
Pred = []
Truth = []
Lead_RMSD_arr = {
    "z": [[] for _ in range(7)],
    "t": [[] for _ in range(7)],
    "t2m": [[] for _ in range(7)],
    "u10": [[] for _ in range(7)],
    "v10": [[] for _ in range(7)],
}
Lead_ACC = {
    "z": [[] for _ in range(7)],
    "t": [[] for _ in range(7)],
    "t2m": [[] for _ in range(7)],
    "u10": [[] for _ in range(7)],
    "v10": [[] for _ in range(7)],
}
Lead_CRPS = {
    "z": [[] for _ in range(7)],
    "t": [[] for _ in range(7)],
    "t2m": [[] for _ in range(7)],
    "u10": [[] for _ in range(7)],
    "v10": [[] for _ in range(7)],
}

for entry, (time_steps, batch) in tqdm(
    enumerate(zip(time_loader, Test_loader)),
    colour="green",
    desc="eval",
    total=len(time_loader),
):
    data = (
        batch[0]
        .to(device)
        .view(num_years, 1, len(paths_to_data) * (args.scale + 1), H, W)
    )
    past_sample = (
        vel_test[entry]
        .view(num_years, 2 * len(paths_to_data) * (args.scale + 1), H, W)
        .to(device)
    )
    model.update_param(
        [
            past_sample,
            const_channels_info.to(device),
            lat_map.to(device),
            lon_map.to(device),
        ]
    )
    t = time_steps.float().to(device).flatten()
    mean_pred, std_pred, mean_wo_bias = model(t, data)
    mean_avg = mean_pred.view(-1, len(paths_to_data) * (args.scale + 1), H, W)
    std_avg = std_pred.view(-1, len(paths_to_data) * (args.scale + 1), H, W)

    for yr in range(2):
        for t_step in range(1, len(time_steps), 1):
            evaluate_rmsd = evaluation_rmsd_mm(
                mean_pred[t_step, yr, :, :, :].cpu(),
                batch[t_step, yr, :, :, :].cpu(),
                lat,
                lon,
                max_lev,
                min_lev,
                H,
                W,
                levels,
            )
            evaluate_acc = evaluation_acc_mm(
                mean_pred[t_step, yr, :, :, :].cpu(),
                batch[t_step, yr, :, :, :].cpu(),
                lat,
                lon,
                max_lev,
                min_lev,
                H,
                W,
                levels,
                clim[yr, :, :, :].cpu().detach().numpy(),
            )
            evaluate_crps = evaluation_crps_mm(
                mean_pred[t_step, yr, :, :, :].cpu(),
                batch[t_step, yr, :, :, :].cpu(),
                lat,
                lon,
                max_lev,
                min_lev,
                H,
                W,
                levels,
                std_pred[t_step, yr, :, :, :].cpu(),
            )
            for idx, lev in enumerate(levels):
                Lead_RMSD_arr[lev][t_step - 1].append(evaluate_rmsd[idx])
                Lead_ACC[lev][t_step - 1].append(evaluate_acc[idx])
                Lead_CRPS[lev][t_step - 1].append(evaluate_crps[idx])


for t_idx in range(8):
    for idx, lev in enumerate(levels):
        print(
            "Lead Time ",
            (t_idx + 1) * 6,
            "hours ",
            "| Observable ",
            lev,
            "| Mean RMSD ",
            np.mean(Lead_RMSD_arr[lev][t_idx]),
            "| Std RMSD ",
            np.std(Lead_RMSD_arr[lev][t_idx]),
        )
        print(
            "Lead Time ",
            (t_idx + 1) * 6,
            "hours ",
            "| Observable ",
            lev,
            "| Mean ACC ",
            np.mean(Lead_ACC[lev][t_idx]),
            "| Std ACC ",
            np.std(Lead_ACC[lev][t_idx]),
        )
        print(
            "Lead Time ",
            (t_idx + 1) * 6,
            "hours ",
            "| Observable ",
            lev,
            "| Mean CRPS ",
            np.mean(Lead_CRPS[lev][t_idx]),
            "| Std CRPS ",
            np.std(Lead_CRPS[lev][t_idx]),
        )
