# %%
import datasets
import xarray as xr
import warnings
from tqdm.cli import tqdm
import os
from torch.utils.data import DataLoader
import torch.nn.functional as Fin
import timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
from torchdiffeq import odeint as odeint
import matplotlib
matplotlib.use('Agg')
import argparse
import sys
import time
import torch
torch.manual_seed(42)
torch.cuda.empty_cache() 
import torch.optim as optim
import random
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)
import sys
from pathlib import Path

# %%

from utils import load_velocity, fit_velocity, get_gauss_kernel
from model_function import Optim_velocity
from model_function import add_constant_info, Climate_encoder_free_uncertain
from utils import count_parameters
from utils import set_seed
from utils import nll

import wandb

set_seed(42)

# %%
cwd = os.getcwd()
#data_path = {'z500':str(cwd) + '/era5_data/geopotential_500/*.nc','t850':str(cwd) + '/era5_data/temperature_850/*.nc'}
SOLVERS = ["dopri8","dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams',"adaptive_heun","euler"]
parser = argparse.ArgumentParser('ClimODE')

# parser.add_argument('--solver', type=str, default="euler", choices=SOLVERS)
# parser.add_argument('--atol', type=float, default=5e-3)
# parser.add_argument('--rtol', type=float, default=5e-3)
# parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")
# parser.add_argument('--niters', type=int, default=300)
# parser.add_argument('--scale', type=int, default=0)
# parser.add_argument('--batch_size', type=int, default=6)
# parser.add_argument('--spectral', type=int, default=0,choices=[0,1])
# parser.add_argument('--lr', type=float, default=0.0005)
# parser.add_argument('--weight_decay', type=float, default=1e-5)

# args = parser.parse_args('--scale 0 --batch_size 6 --spectral 0 --solver euler'.split())
args = argparse.Namespace(
    solver='euler',
    atol=5e-3,
    rtol=5e-3,
    step_size=None,
    niters=300,
    scale=0,
    batch_size=6,
    spectral=0,
    lr=0.0005,
    weight_decay=1e-5,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args

# %%
train_time_scale= slice('2006','2016')
val_time_scale = slice('2016','2016')
test_time_scale = slice('2017','2018')

paths_to_data = [str(cwd) + '/era5_data/geopotential_500/*.nc',
                 str(cwd) + '/era5_data/temperature_850/*.nc',
                 str(cwd) + '/era5_data/2m_temperature/*.nc',
                 str(cwd) + '/era5_data/10m_u_component_of_wind/*.nc',
                 str(cwd) + '/era5_data/10m_v_component_of_wind/*.nc']
const_info_path = [str(cwd) +  '/era5_data/constants/constants/constants_5.625deg.nc']
levels = ["z","t","t2m","u10","v10"]
paths_to_data = paths_to_data[0:5]
levels = levels[0:5]
assert len(paths_to_data) == len(levels), "Paths to different type of data must be same as number of types of observations"

# %%

def get_batched(train_times,data_train_final,lev):
    for idx,year in enumerate(train_times):
        data_per_year = data_train_final.sel(time=slice(str(year),str(year))).load()
        data_values = data_per_year[lev].values
        if idx ==0:
            train_data = torch.from_numpy(data_values).reshape(-1,1,1,data_values.shape[-2],data_values.shape[-1])
            if year%4==0: train_data = torch.cat((train_data[:236],train_data[240:])) #skipping 29 feb in leap year
        else:
            mid_data = torch.from_numpy(data_values).reshape(-1,1,1,data_values.shape[-2],data_values.shape[-1])
            if year%4==0: mid_data = torch.cat((mid_data[:236],mid_data[240:]))#skipping 29 feb in leap year
            train_data = torch.cat([train_data,mid_data],dim=1)
    
    return train_data

def get_train_test_data_without_scales_batched(data_path,train_time_scale,val_time_scale,test_time_scale,lev,spectral):
    data = xr.open_mfdataset(data_path, combine='by_coords')
    #data = data.isel(lat=slice(None, None, -1))
    if lev in ["v","u","r","q","tisr"]:
        data = data.sel(level=500)
    data = data.resample(time="6H").nearest(tolerance="1H") # Setting data to be 6-hour cycles
    data_train = data.sel(time=train_time_scale).load()
    data_val = data.sel(time=val_time_scale).load()
    data_test = data.sel(time=test_time_scale).load()
    data_global = data.sel(time=slice('2006','2018')).load()

    max_val = data_global.max()[lev].values.tolist()
    min_val = data_global.min()[lev].values.tolist()


    data_train_final = (data_train - min_val)/ (max_val - min_val)
    data_val_final = (data_val - min_val)/ (max_val - min_val)
    data_test_final = (data_test - min_val)/ (max_val - min_val)

    time_vals = data_test_final.time.values
    train_times = [i for i in range(2006,2016)]
    test_times = [2017,2018]
    val_times = [2016]

    train_data = get_batched(train_times,data_train_final,lev)
    test_data = get_batched(test_times,data_test_final,lev)
    val_data = get_batched(val_times,data_val_final,lev)

    t = [i for i in range(365*4)]
    time_steps = torch.tensor(t).view(-1,1)
    return train_data,val_data,test_data,time_steps,data.lat.values,data.lon.values,max_val,min_val,time_vals

# %%
Final_train_data = 0
Final_val_data = 0
Final_test_data = 0
max_lev = []
min_lev = []

for idx,data in enumerate(paths_to_data):
    Train_data,Val_data,Test_data,time_steps,lat,lon,mean,std,time_stamp = \
        get_train_test_data_without_scales_batched(data,train_time_scale,val_time_scale,test_time_scale,levels[idx],args.spectral)  
    max_lev.append(mean)
    min_lev.append(std)
    if idx==0: 
        Final_train_data = Train_data
        Final_val_data = Val_data
        Final_test_data = Test_data
    else:
        Final_train_data = torch.cat([Final_train_data,Train_data],dim=2)
        Final_val_data = torch.cat([Final_val_data,Val_data],dim=2)
        Final_test_data = torch.cat([Final_test_data,Test_data],dim=2)


print("Length of training data",len(Final_train_data))
print("Length of validation data",len(Final_val_data))
print("Length of testing data",len(Final_test_data))

# %%

const_channels_info,lat_map,lon_map = add_constant_info(const_info_path)
H,W = Train_data.shape[3],Train_data.shape[4]
Train_loader = DataLoader(Final_train_data[2:],
                          batch_size=args.batch_size,shuffle=False,pin_memory=False)
Val_loader = DataLoader(Final_val_data[2:],
                        batch_size=args.batch_size,shuffle=False,pin_memory=False)
Test_loader = DataLoader(Final_test_data[2:],
                         batch_size=args.batch_size,shuffle=False,pin_memory=False)
time_loader = DataLoader(time_steps[2:],
                         batch_size=args.batch_size,shuffle=False,pin_memory=False)
time_idx_steps = torch.tensor([i for i in range(365*4)]).view(-1,1)
time_idx = DataLoader(time_idx_steps[2:],batch_size=args.batch_size,shuffle=False,pin_memory=False)
#Model declaration
num_years = len(range(2006,2016))
model = Climate_encoder_free_uncertain(len(paths_to_data),2,out_types=len(paths_to_data),method=args.solver,use_att=True,use_err=True,use_pos=False).to(device)
#model.apply(weights_init_uniform_rule)

param = count_parameters(model)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300)

best_loss = float('inf')
train_best_loss = float('inf')
best_epoch = float('inf')

# %%
if not Path(str(cwd) +"/kernel.npy").exists():
    get_gauss_kernel((32,64),lat,lon)
else:
    print("## DETECTED kernel.npy from previous run, will load from disk instead of regen. ")

kernel = torch.from_numpy(np.load(str(cwd) +"/kernel.npy"))
#breakpoint()
if not Path('test_10year_2day_mm_vel.npy').exists():
    print("Fitting velocity...")
    fit_velocity(time_idx,time_loader,Final_train_data,Train_loader,torch.device('cuda'),num_years,paths_to_data,args.scale,H,W,types='train_10year_2day_mm',vel_model=Optim_velocity,kernel=kernel,lat=lat,lon=lon)
    fit_velocity(time_idx,time_loader,Final_val_data,Val_loader,torch.device('cuda'),1,paths_to_data,args.scale,H,W,types='val_10year_2day_mm',vel_model=Optim_velocity,kernel=kernel,lat=lat,lon=lon)
    fit_velocity(time_idx,time_loader,Final_test_data,Test_loader,torch.device('cuda'),2,paths_to_data,args.scale,H,W,types='test_10year_2day_mm',vel_model=Optim_velocity,kernel=kernel,lat=lat,lon=lon)
else:
    print("## DETECTED test_10year_2day_mm_vel.npy from previous run, will load from disk instead of regen")
    
vel_train,vel_val = load_velocity(['train_10year_2day_mm','val_10year_2day_mm'])

torch.cuda.empty_cache()

print("############################ Velocity loaded, Model starts to train #########################")
print(model)
print("####################### Total Parameters",param ,"################################")

# %%
print(f"Model has {param:.1e} trainable parameters")
# %%
import wandb

# Initialize wandb
wandb.init(
    project="climode",
    entity='tornikeo1',
    config={
        **vars(args),
        "params": param,
        "architecture": "default",
        "dataset": "TornikeO/era5-5.625deg",
    }
)

best_loss = float('inf')
best_epoch = 0

for epoch in range(args.niters):
    print(f"##### Epoch {epoch} of {args.niters} #####")
    total_train_loss = 0
    val_loss = 0

    if epoch == 0:
        var_coeff = 0.001
    else:
        var_coeff = 2 * scheduler.get_last_lr()[0]

    _total = min(len(time_loader), len(Train_loader))
    pbar = tqdm(enumerate(zip(time_loader, Train_loader)), total=_total, colour='green', desc='train')

    for entry, (time_steps, batch) in pbar:
        optimizer.zero_grad()
        data = batch[0].to(device).view(num_years, 1, len(paths_to_data) * (args.scale + 1), H, W)
        past_sample = vel_train[entry].view(num_years, 2 * len(paths_to_data) * (args.scale + 1), H, W).to(device)
        model.update_param([past_sample, const_channels_info.to(device), lat_map.to(device), lon_map.to(device)])
        t = time_steps.float().to(device).flatten()
        mean, std, _ = model(t, data)
        loss = nll(mean, std, batch.float().to(device), lat, var_coeff)

        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = loss + l2_lambda * l2_norm
        loss.backward()
        optimizer.step()

        if torch.isnan(loss):
            print("Quitting due to NaN loss")
            quit()

        total_train_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
        wandb.log({"train_loss": loss.item()})
        
    lr_val = scheduler.get_last_lr()[0]
    scheduler.step()

    print(f"|Iter {epoch} | Total Train Loss {total_train_loss}|")
    optimizer.zero_grad(set_to_none=True)  # Clear memory
    torch.cuda.empty_cache()

    with torch.no_grad():
        pbar = tqdm(enumerate(zip(time_loader, Val_loader)), total=min(len(time_loader), len(Val_loader)), colour='blue', desc='test')
        for entry, (time_steps, batch) in pbar:
            data = batch[0].to(device).view(1, 1, len(paths_to_data) * (args.scale + 1), H, W)
            past_sample = vel_val[entry].view(1, 2 * len(paths_to_data) * (args.scale + 1), H, W).to(device)
            model.update_param([past_sample, const_channels_info.to(device), lat_map.to(device), lon_map.to(device)])
            t = time_steps.float().to(device).flatten()
            mean, std, _ = model(t, data)
            loss = nll(mean, std, batch.float().to(device), lat, var_coeff)

            if torch.isnan(loss):
                print("Quitting due to NaN loss")
                quit()

            val_loss += loss.item()
            wandb.log({"val_loss": loss.item()})
            pbar.set_postfix({"val_loss": loss.item()})

    print(f"|Iter {epoch} | Total Val Loss {val_loss}|")

    # Log metrics to wandb
    wandb.log({
        "epoch": epoch,
        "total_train_loss": total_train_loss,
        "total_val_loss": val_loss,
        "learning_rate": lr_val,
        "var_coeff": var_coeff,
    })

    # Save the best model
    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = epoch
        saved_model = f"checkpoints/ClimODE_global_{args.solver}_{args.spectral}_model_{epoch}_{val_loss:.4f}.pt"
        torch.save(model, saved_model)
        wandb.save(saved_model)
        
    torch.cuda.empty_cache()
    
wandb.finish()