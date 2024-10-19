import argparse
import data.ts_datasets as my
import pandas as pd
from models import DumbMLP4TS, ARv2, ARv2_normal, ARv2_search
import torch
import numpy as np
import pickle as pkl
import time 
import sys
from scipy.stats import boxcox
import warnings
import copy
import math
from tqdm import tqdm

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

def _unfold_df(df, device, input_length, output_length, for_var=False):
    data = torch.tensor(df.to_numpy() if for_var else np.transpose(df.to_numpy()), dtype=torch.float64 if for_var else torch.float64, device=device)
    return data.unfold(0 if for_var else 1, input_length + output_length, 1)

def main(dataset):
    dataset = dataset
    print("=================================starting ARIMA on dataset:", dataset, "=================================")
    batch_size = 64
    multi_channel = False

    file_dict = {
            "ETTh1": "ETT-small/ETTh1.csv",
            "ETTh2": "ETT-small/ETTh2.csv",
            "ETTm1": "ETT-small/ETTm1.csv",
            "ETTm2": "ETT-small/ETTm2.csv",
            "ECL": "electricity/electricity.csv",
            "ER": "exchange_rate/exchange_rate.csv",
            "ILI": "illness/national_illness.csv",
            "Traffic": "traffic/traffic.csv",
            "Weather": "weather/weather.csv"
        }

        
    params = dataset.split("_")
    prefix = params[0]
    horizon = int(params[1])
    horizon = horizon
    assert prefix in file_dict, f"Invalid dataset {dataset} from possible {list(file_dict.keys())}"
    input_length = 96 if prefix == "ILI" else 512
    train_loader, val_loader, test_loader = my.get_timeseries_dataloaders(
        f"./data/ts_datasets/all_six_datasets/{file_dict[prefix]}", 
        batch_size=batch_size, 
        seq_len=input_length,
        forecast_horizon=horizon,
        multi_channel=multi_channel
    )
    train_df, val_df, test_df = train_loader.dataset.data, val_loader.dataset.data, test_loader.dataset.data
    print(train_df.shape, val_df.shape, test_df.shape)
    train_val_df = np.concatenate((train_df, val_df), axis=0)
    test_df = pd.DataFrame(test_df)
    device = torch.device("cpu")
    test_series_rw = _unfold_df(test_df, device, input_length, horizon)
    test_true = test_series_rw[:, :, -horizon:]
    test_x = test_series_rw[:, :, :-horizon]

    mse = []
    for i in range(train_df.shape[1]):
        print("ARIMA for channel", i)
        ap = train_val_df[:, i]
        print(ap.shape)
        arima = AutoARIMA()
        arima = arima.fit(y=ap)
        for j in range(test_series_rw.shape[1]):
            y_new = np.array(test_x[i, j, :]).reshape(-1,)
            y_pred = arima.forward(y=y_new, h=horizon)["mean"]
            mse.append(np.mean((y_pred - test_true[i, j, :].numpy())**2))
        print(mse[-1])

    print("MSE:", np.mean(mse))


if __name__ == "__main__":
    dataset = ["ETTm1_96", "ETTm1_192", "ETTm1_336", "ETTm1_720"]
    for i in dataset:
        main(i)