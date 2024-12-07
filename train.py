import torch
import os
import datasets
import warnings
from tqdm.cli import tqdm
import os
from torch.utils.data import DataLoader
import torch.nn.functional as Fin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from torchdiffeq import odeint as odeint
import matplotlib
import argparse
import torch
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import xarray as xr


def get_batched(train_times, data_train_final, lev):
    for idx, year in enumerate(train_times):
        data_per_year = data_train_final.sel(time=slice(str(year), str(year))).load()
        data_values = data_per_year[lev].values
        if idx == 0:
            train_data = torch.from_numpy(data_values).reshape(
                -1, 1, 1, data_values.shape[-2], data_values.shape[-1]
            )
            if year % 4 == 0:
                train_data = torch.cat(
                    (train_data[:236], train_data[240:])
                )  # skipping 29 feb in leap year
        else:
            mid_data = torch.from_numpy(data_values).reshape(
                -1, 1, 1, data_values.shape[-2], data_values.shape[-1]
            )
            if year % 4 == 0:
                mid_data = torch.cat(
                    (mid_data[:236], mid_data[240:])
                )  # skipping 29 feb in leap year
            train_data = torch.cat([train_data, mid_data], dim=1)

    return train_data


def get_train_test_data_without_scales_batched(
    data_path, train_time_scale, val_time_scale, test_time_scale, lev, spectral
):
    data = xr.open_mfdataset(data_path, combine="by_coords")
    # data = data.isel(lat=slice(None, None, -1))
    if lev in ["v", "u", "r", "q", "tisr"]:
        data = data.sel(level=500)
    data = data.resample(time="6h").nearest(
        tolerance="1h"
    )  # Setting data to be 6-hour cycles
    data_train = data.sel(time=train_time_scale).load()
    data_val = data.sel(time=val_time_scale).load()
    data_test = data.sel(time=test_time_scale).load()
    data_global = data.sel(time=slice("2006", "2018")).load()

    max_val = data_global.max()[lev].values.tolist()
    min_val = data_global.min()[lev].values.tolist()

    data_train_final = (data_train - min_val) / (max_val - min_val)
    data_val_final = (data_val - min_val) / (max_val - min_val)
    data_test_final = (data_test - min_val) / (max_val - min_val)

    time_vals = data_test_final.time.values
    train_times = [i for i in range(2006, 2016)]
    test_times = [2017, 2018]
    val_times = [2016]

    train_data = get_batched(train_times, data_train_final, lev)
    test_data = get_batched(test_times, data_test_final, lev)
    val_data = get_batched(val_times, data_val_final, lev)

    t = [i for i in range(365 * 4)]
    time_steps = torch.tensor(t).view(-1, 1)
    return (
        train_data,
        val_data,
        test_data,
        time_steps,
        data.lat.values,
        data.lon.values,
        max_val,
        min_val,
        time_vals,
    )


def load_velocity(types):
    cwd = os.getcwd()
    vel = []
    for file in types:
        vel.append(np.load(str(cwd) + "/" + file + "_vel.npy"))

    return (torch.from_numpy(v) for v in vel)


def add_constant_info(path):
    data = xr.open_mfdataset(path, combine="by_coords")
    for idx, var in enumerate(["orography", "lsm"]):
        var_value = torch.from_numpy(data[var].values).view(1, 1, 32, 64)
        if idx == 0:
            final_var = var_value
        else:
            final_var = torch.cat([final_var, var_value], dim=1)

    return (
        final_var,
        torch.from_numpy(data["lat2d"].values),
        torch.from_numpy(data["lon2d"].values),
    )


class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
    ):
        super().__init__()
        self.activation = nn.LeakyReLU(0.3)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), padding=0
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(p=0.1)
        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        # First convolution layer
        x_mod = F.pad(F.pad(x, (0, 0, 1, 1), "reflect"), (1, 1, 0, 0), "circular")
        h = self.activation(self.bn1(self.conv1(self.norm1(x_mod))))
        # Second convolution layer
        h = F.pad(F.pad(h, (0, 0, 1, 1), "reflect"), (1, 1, 0, 0), "circular")
        h = self.activation(self.bn2(self.conv2(self.norm2(h))))
        h = self.drop(h)
        # Add the shortcut connection and return
        return h + self.shortcut(x)


class Climate_ResNet_2D(nn.Module):

    def __init__(self, num_channels, layers, hidden_size):
        super().__init__()
        layers_cnn = []
        activation_fns = []
        self.block = ResidualBlock
        self.inplanes = num_channels

        for idx in range(len(layers)):
            if idx == 0:
                layers_cnn.append(
                    self.make_layer(
                        self.block, num_channels, hidden_size[idx], layers[idx]
                    )
                )
            else:
                layers_cnn.append(
                    self.make_layer(
                        self.block, hidden_size[idx - 1], hidden_size[idx], layers[idx]
                    )
                )

        self.layer_cnn = nn.ModuleList(layers_cnn)
        self.activation_cnn = nn.ModuleList(activation_fns)

    def make_layer(self, block, in_channels, out_channels, reps):
        layers = []
        layers.append(block(in_channels, out_channels))
        self.inplanes = out_channels
        for i in range(1, reps):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, data):
        dx_final = data.float()
        for l, layer in enumerate(self.layer_cnn):
            dx_final = layer(dx_final)

        return dx_final


class boundarypad(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return F.pad(F.pad(input, (0, 0, 1, 1), "reflect"), (1, 1, 0, 0), "circular")


class Self_attn_conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Self_attn_conv, self).__init__()
        self.query = self._conv(in_channels, in_channels // 8, stride=1)
        self.key = self.key_conv(in_channels, in_channels // 8, stride=2)
        self.value = self.key_conv(in_channels, out_channels, stride=2)
        self.post_map = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels, kernel_size=(1, 1), stride=1, padding=0
            )
        )
        self.out_ch = out_channels

    def _conv(self, n_in, n_out, stride):
        return nn.Sequential(
            boundarypad(),
            nn.Conv2d(n_in, n_in // 2, kernel_size=(3, 3), stride=stride, padding=0),
            nn.LeakyReLU(0.3),
            boundarypad(),
            nn.Conv2d(n_in // 2, n_out, kernel_size=(3, 3), stride=stride, padding=0),
            nn.LeakyReLU(0.3),
            boundarypad(),
            nn.Conv2d(n_out, n_out, kernel_size=(3, 3), stride=stride, padding=0),
        )

    def key_conv(self, n_in, n_out, stride):
        return nn.Sequential(
            nn.Conv2d(n_in, n_in // 2, kernel_size=(3, 3), stride=stride, padding=0),
            nn.LeakyReLU(0.3),
            nn.Conv2d(n_in // 2, n_out, kernel_size=(3, 3), stride=stride, padding=0),
            nn.LeakyReLU(0.3),
            nn.Conv2d(n_out, n_out, kernel_size=(3, 3), stride=1, padding=0),
        )

    def forward(self, x):
        size = x.size()
        x = x.float()
        q, k, v = (
            self.query(x).flatten(-2, -1),
            self.key(x).flatten(-2, -1),
            self.value(x).flatten(-2, -1),
        )
        beta = F.softmax(torch.bmm(q.transpose(1, 2), k), dim=1)
        o = torch.bmm(v, beta.transpose(1, 2))
        o = self.post_map(o.view(-1, self.out_ch, size[-2], size[-1]).contiguous())
        return o


class Climate_encoder_free_uncertain(nn.Module):

    def __init__(
        self, num_channels, const_channels, out_types, method, use_att, use_err, use_pos
    ):
        super().__init__()
        self.layers = [5, 3, 2]
        self.hidden = [128, 64, 2 * out_types]
        input_channels = 30 + out_types * int(use_pos) + 34 * (1 - int(use_pos))
        self.vel_f = Climate_ResNet_2D(input_channels, self.layers, self.hidden)

        assert use_att
        self.vel_att = Self_attn_conv(input_channels, 10)
        self.gamma = nn.Parameter(torch.tensor([0.1]))

        self.scales = num_channels
        self.const_channel = const_channels

        self.out_ch = out_types
        self.past_samples = 0
        self.const_info = 0
        self.lat_map = 0
        self.lon_map = 0
        self.elev = 0
        self.pos_emb = 0
        self.elev_info_grad_x = 0
        self.elev_info_grad_y = 0
        self.method = method
        err_in = 9 + out_types * int(use_pos) + 34 * (1 - int(use_pos))
        assert use_err
        self.noise_net = Climate_ResNet_2D(
            err_in, [3, 2, 2], [128, 64, 2 * out_types]
        )
        self.att = use_att
        self.err = use_err
        self.pos = use_pos
        self.pos_feat = 0
        self.lsm = 0
        self.oro = 0

    def update_param(self, params):
        self.past_samples = params[0]
        self.const_info = params[1]
        self.lat_map = params[2]
        self.lon_map = params[3]

    def pde(self, t, vs):

        ds = (
            vs[:, -self.out_ch :, :, :]
            .view(-1, self.out_ch, vs.shape[2], vs.shape[3])
            .float()
        )
        v = (
            vs[:, : 2 * self.out_ch, :, :]
            .view(-1, 2 * self.out_ch, vs.shape[2], vs.shape[3])
            .float()
        )
        t_emb = (
            ((t * 100) % 24)
            .view(1, 1, 1, 1)
            .expand(ds.shape[0], 1, ds.shape[2], ds.shape[3])
        )
        sin_t_emb = torch.sin(torch.pi * t_emb / 12 - torch.pi / 2)
        cos_t_emb = torch.cos(torch.pi * t_emb / 12 - torch.pi / 2)

        sin_seas_emb = torch.sin(torch.pi * t_emb / (12 * 365) - torch.pi / 2)
        cos_seas_emb = torch.cos(torch.pi * t_emb / (12 * 365) - torch.pi / 2)

        day_emb = torch.cat([sin_t_emb, cos_t_emb], dim=1)
        seas_emb = torch.cat([sin_seas_emb, cos_seas_emb], dim=1)

        ds_grad_x = torch.gradient(ds, dim=3)[0]
        ds_grad_y = torch.gradient(ds, dim=2)[0]
        nabla_u = torch.cat([ds_grad_x, ds_grad_y], dim=1)

        assert not self.pos
            
        cos_lat_map, sin_lat_map = torch.cos(self.new_lat_map), torch.sin(
            self.new_lat_map
        )
        cos_lon_map, sin_lon_map = torch.cos(self.new_lon_map), torch.sin(
            self.new_lon_map
        )
        t_cyc_emb = torch.cat([day_emb, seas_emb], dim=1)
        pos_feats = torch.cat(
            [
                cos_lat_map,
                cos_lon_map,
                sin_lat_map,
                sin_lon_map,
                sin_lat_map * cos_lon_map,
                sin_lat_map * sin_lon_map,
            ],
            dim=1,
        )
        pos_time_ft = self.get_time_pos_embedding(t_cyc_emb, pos_feats)
        comb_rep = torch.cat(
            [
                t_emb / 24,
                day_emb,
                seas_emb,
                nabla_u,
                v,
                ds,
                self.new_lat_map,
                self.new_lon_map,
                self.lsm,
                self.oro,
                pos_feats,
                pos_time_ft,
            ],
            dim=1,
        )

        assert self.att
        dv = self.vel_f(comb_rep) + self.gamma * self.vel_att(comb_rep)
        v_x = (
            v[:, : self.out_ch, :, :]
            .view(-1, self.out_ch, vs.shape[2], vs.shape[3])
            .float()
        )
        v_y = (
            v[:, -self.out_ch :, :, :]
            .view(-1, self.out_ch, vs.shape[2], vs.shape[3])
            .float()
        )

        adv1 = v_x * ds_grad_x + v_y * ds_grad_y
        adv2 = ds * (torch.gradient(v_x, dim=3)[0] + torch.gradient(v_y, dim=2)[0])

        ds = adv1 + adv2

        dvs = torch.cat([dv, ds], 1)
        return dvs

    def get_time_pos_embedding(self, time_feats, pos_feats):
        for idx in range(time_feats.shape[1]):
            tf = time_feats[:, idx].unsqueeze(dim=1) * pos_feats
            if idx == 0:
                final_out = tf
            else:
                final_out = torch.cat([final_out, tf], dim=1)

        return final_out

    def noise_net_contrib(self, t, pos_enc, s_final, noise_net, H, W):

        t_emb = (t % 24).view(-1, 1, 1, 1, 1)
        sin_t_emb = torch.sin(torch.pi * t_emb / 12 - torch.pi / 2).expand(
            len(s_final), s_final.shape[1], 1, H, W
        )
        cos_t_emb = torch.cos(torch.pi * t_emb / 12 - torch.pi / 2).expand(
            len(s_final), s_final.shape[1], 1, H, W
        )

        sin_seas_emb = torch.sin(torch.pi * t_emb / (12 * 365) - torch.pi / 2).expand(
            len(s_final), s_final.shape[1], 1, H, W
        )
        cos_seas_emb = torch.cos(torch.pi * t_emb / (12 * 365) - torch.pi / 2).expand(
            len(s_final), s_final.shape[1], 1, H, W
        )

        pos_enc = pos_enc.expand(len(s_final), s_final.shape[1], -1, H, W).flatten(
            start_dim=0, end_dim=1
        )
        t_cyc_emb = torch.cat(
            [sin_t_emb, cos_t_emb, sin_seas_emb, cos_seas_emb], dim=2
        ).flatten(start_dim=0, end_dim=1)

        pos_time_ft = self.get_time_pos_embedding(t_cyc_emb, pos_enc[:, 2:-2])

        comb_rep = torch.cat(
            [t_cyc_emb, s_final.flatten(start_dim=0, end_dim=1), pos_enc, pos_time_ft],
            dim=1,
        )

        final_out = noise_net(comb_rep).view(len(t), -1, 2 * self.out_ch, H, W)

        mean = s_final + final_out[:, :, : self.out_ch]
        std = nn.Softplus()(final_out[:, :, self.out_ch :])

        return mean, std

    def forward(self, T, data, atol=0.1, rtol=0.1):
        H, W = self.past_samples.shape[2], self.past_samples.shape[3]
        final_data = torch.cat(
            [self.past_samples, data.float().view(-1, self.out_ch, H, W)], 1
        )
        init_time = T[0].item() * 6
        final_time = T[-1].item() * 6
        steps_val = final_time - init_time

        # breakpoint()
        assert not self.pos
        self.oro, self.lsm = self.const_info[0, 0], self.const_info[0, 1]
        self.lsm = self.lsm.unsqueeze(dim=0).expand(
            data.shape[0], -1, data.shape[3], data.shape[4]
        )
        self.oro = (
            F.normalize(self.const_info[0, 0])
            .unsqueeze(dim=0)
            .expand(data.shape[0], -1, data.shape[3], data.shape[4])
        )
        self.new_lat_map = (
            self.lat_map.expand(data.shape[0], 1, data.shape[3], data.shape[4])
            * torch.pi
            / 180
        )  # Converting to radians
        self.new_lon_map = (
            self.lon_map.expand(data.shape[0], 1, data.shape[3], data.shape[4])
            * torch.pi
            / 180
        )
        cos_lat_map, sin_lat_map = torch.cos(self.new_lat_map), torch.sin(
            self.new_lat_map
        )
        cos_lon_map, sin_lon_map = torch.cos(self.new_lon_map), torch.sin(
            self.new_lon_map
        )
        pos_feats = torch.cat(
            [
                cos_lat_map,
                cos_lon_map,
                sin_lat_map,
                sin_lon_map,
                sin_lat_map * cos_lon_map,
                sin_lat_map * sin_lon_map,
            ],
            dim=1,
        )
        final_pos_enc = torch.cat(
            [self.new_lat_map, self.new_lon_map, pos_feats, self.lsm, self.oro],
            dim=1,
        )

        new_time_steps = torch.linspace(
            init_time, final_time, steps=int(steps_val) + 1
        ).to(data.device)
        t = 0.01 * new_time_steps.float().to(data.device).flatten().float()
        pde_rhs = lambda t, vs: self.pde(t, vs)  # make the ODE forward function
        final_result = odeint(
            pde_rhs, final_data, t, method=self.method, atol=atol, rtol=rtol
        )
        # breakpoint()
        s_final = final_result[:, :, -self.out_ch :, :, :].view(
            len(t), -1, self.out_ch, H, W
        )

        assert self.err
        mean, std = self.noise_net_contrib(
            T, final_pos_enc, s_final[0 : len(s_final) : 6], self.noise_net, H, W
        )

        return mean, std, s_final[0 : len(s_final) : 6]


def nll(mean, std, truth, lat, var_coeff):
    normal_lkl = torch.distributions.normal.Normal(mean, 1e-3 + std)
    lkl = -normal_lkl.log_prob(truth)
    loss_val = lkl.mean() + var_coeff * (std**2).sum()
    # loss_val = torch.mean(lkl,dim=(0,1,3,4))
    return loss_val


def main():
    torch.manual_seed(42)

    cwd = os.getcwd()
    # data_path = {'z500':str(cwd) + '/era5_data/geopotential_500/*.nc','t850':str(cwd) + '/era5_data/temperature_850/*.nc'}
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
        "euler",
    ]
    parser = argparse.ArgumentParser("ClimODE")

    parser.add_argument("--solver", type=str, default="euler", choices=SOLVERS)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument(
        "--step_size", type=float, default=None, help="Optional fixed step size."
    )
    parser.add_argument("--niters", type=int, default=300)
    parser.add_argument("--scale", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--spectral", type=int, default=0, choices=[0, 1])
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    args = parser.parse_args(
        "--scale 0 --batch_size 6 --spectral 0 --solver euler".split()
    )
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    args

    # %%
    train_time_scale = slice("2006", "2016")
    val_time_scale = slice("2016", "2016")
    test_time_scale = slice("2017", "2018")
    paths_to_data = [
        "era5_data/geopotential_500/*.nc",
        "era5_data/temperature_850/*.nc",
        "era5_data/2m_temperature/*.nc",
        "era5_data/10m_u_component_of_wind/*.nc",
        "era5_data/10m_v_component_of_wind/*.nc",
    ]
    const_info_path = ["era5_data/constants/constants/constants_5.625deg.nc"]
    levels = ["z", "t", "t2m", "u10", "v10"]

    assert len(paths_to_data) == len(
        levels
    ), "Paths to different type of data must be same as number of types of observations"

    Final_train_data = 0
    Final_val_data = 0
    Final_test_data = 0
    max_lev = []
    min_lev = []

    for idx, data in enumerate(tqdm(paths_to_data, desc="reading data")):
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

    print("train, val, test data shapes:")
    print(Final_train_data.shape, Final_test_data.shape, Final_val_data.shape)

    kernel = torch.from_numpy(np.load(str(cwd) + "/kernel.npy"))
    vel_train, vel_val = load_velocity(["train_10year_2day_mm", "val_10year_2day_mm"])

    const_channels_info, lat_map, lon_map = add_constant_info(const_info_path)
    H, W = Train_data.shape[3], Train_data.shape[4]
    Train_loader = DataLoader(
        Final_train_data[2:],
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False,
    )
    Val_loader = DataLoader(
        Final_val_data[2:], batch_size=args.batch_size, shuffle=False, pin_memory=False
    )
    Test_loader = DataLoader(
        Final_test_data[2:], batch_size=args.batch_size, shuffle=False, pin_memory=False
    )
    time_loader = DataLoader(
        time_steps[2:], batch_size=args.batch_size, shuffle=False, pin_memory=False
    )
    time_idx_steps = torch.tensor([i for i in range(365 * 4)]).view(-1, 1)
    time_idx = DataLoader(
        time_idx_steps[2:], batch_size=args.batch_size, shuffle=False, pin_memory=False
    )

    # Model declaration
    num_years = len(range(2006, 2016))
    model = Climate_encoder_free_uncertain(
        len(paths_to_data),
        2,
        out_types=len(paths_to_data),
        method=args.solver,
        use_att=True,
        use_err=True,
        use_pos=False,
    ).to(device)

    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300)

    best_loss = float("inf")
    train_best_loss = float("inf")
    best_epoch = float("inf")

    for epoch in range(args.niters):
        print(f"##### Epoch {epoch} of {args.niters} #####")
        total_train_loss = 0
        val_loss = 0
        test_loss = 0
        # RMSD = []
        # breakpoint()
        if epoch == 0:
            var_coeff = 0.001
        else:
            var_coeff = 2 * scheduler.get_last_lr()[0]

        _total = min(len(time_loader), len(Train_loader))
        pbar = tqdm(
            enumerate(zip(time_loader, Train_loader)),
            total=_total,
            colour="green",
            desc="train",
        )
        for entry, (time_steps, batch) in pbar:
            optimizer.zero_grad()
            data = (
                batch[0]
                .to(device)
                .view(num_years, 1, len(paths_to_data) * (args.scale + 1), H, W)
            )
            past_sample = (
                vel_train[entry]
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
            mean, std, _ = model(t, data)
            loss = nll(mean, std, batch.float().to(device), lat, var_coeff)
            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm
            loss.backward()
            optimizer.step()
            # print("Loss for batch is ",loss.item())
            pbar.set_postfix({"loss": loss.item()})
            if torch.isnan(loss):
                print("Quitting due to Nan loss")
                quit()
            total_train_loss = total_train_loss + loss.item()

            break

        lr_val = scheduler.get_last_lr()[0]
        scheduler.step()
        print("|Iter ", epoch, " | Total Train Loss ", total_train_loss, "|")
        optimizer.zero_grad(set_to_none=True)  # Clear memory
        torch.cuda.empty_cache()

        with torch.no_grad():
            pbar = tqdm(
                enumerate(zip(time_loader, Val_loader)),
                total=min(len(time_loader), len(Val_loader)),
                colour="blue",
                desc="test",
            )
            for entry, (time_steps, batch) in pbar:
                data = (
                    batch[0]
                    .to(device)
                    .view(1, 1, len(paths_to_data) * (args.scale + 1), H, W)
                )
                past_sample = (
                    vel_val[entry]
                    .view(1, 2 * len(paths_to_data) * (args.scale + 1), H, W)
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
                mean, std, _ = model(t, data)
                loss = nll(mean, std, batch.float().to(device), lat, var_coeff)
                if torch.isnan(loss):
                    print("Quitting due to Nan loss")
                    quit()
                pbar.set_postfix({"val_lss": loss.item()})
                val_loss = val_loss + loss.item()
                break

        print("|Iter ", epoch, " | Total Val Loss ", val_loss, "|")

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(
                model,
                str(cwd)
                + "/Models/"
                + "ClimODE_global_"
                + args.solver
                + "_"
                + str(args.spectral)
                + "_model_"
                + str(epoch)
                + ".pt",
            )
        break


if __name__ == "__main__":
    main()
