import argparse
import data.ts_datasets as tsdata
import pandas as pd
from models import AR_normal, AR_diff
import torch
import numpy as np
import pickle as pkl
import time 
import sys
from scipy.stats import boxcox
import warnings
import copy
import math

def main():
    parser = argparse.ArgumentParser(description='Run AR search')
    parser.add_argument('--dataset', type=str, default='ETTh1_96', help='dataset to use')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--multi_channel', type=bool, default=False, help='use multi-channel data')
    parser.add_argument('--time_limit_hours', type=float, default=10, help='time limit in hours')
    parser.add_argument('--use_ols', type=bool, default=True, help='use OLS for AR')
    parser.add_argument('--kpss', action='store_true', help='do kpss test')
    parser.add_argument('--new_metric', action='store_true', help='use new metric')
    parser.add_argument('--adf', action='store_true', help='use adf test')
    args = parser.parse_args()
    print("=================================starting AR on dataset:", args.dataset, "=================================")

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("Using GPU")
    else:
        DEVICE = torch.device("cpu")
        print("Using CPU")

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

    
    params = args.dataset.split("_")
    prefix = params[0]
    horizon = int(params[1])
    args.horizon = horizon
    assert prefix in file_dict, f"Invalid dataset {args.dataset} from possible {list(file_dict.keys())}"
    input_length = 96 if prefix == "ILI" else 512
    train_loader, val_loader, test_loader = tsdata.get_timeseries_dataloaders(
        f"./data/ts_datasets/all_six_datasets/{file_dict[prefix]}", 
        batch_size=args.batch_size, 
        seq_len=input_length,
        forecast_horizon=horizon,
        multi_channel=args.multi_channel
    )
    train_df, val_df, test_df = train_loader.dataset.data, val_loader.dataset.data, test_loader.dataset.data

    del train_loader, val_loader, test_loader
    if args.new_metric:
        train_val_df = np.concatenate((train_df, val_df), axis=0)
        if args.kpss:
            from statsmodels.tsa.stattools import kpss
            numdiff = 0
            stable = False
            train_val_df_copy = copy.deepcopy(train_val_df)
            print("-----------------KPSS on training -------------------")
            while stable == False:
                print(f"Trying {numdiff} differencing")
                train_val_df = np.diff(train_val_df_copy, n = numdiff, axis = 0)
                pas = True
                for i in range(train_val_df.shape[1]):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        kpss_stat = kpss(train_val_df[:,i], regression='ct')
                    if kpss_stat[1] <= 0.05:
                        print(f"KPSS test failed for {numdiff} differencing on {i}th column")
                        pas = False
                    if i == train_val_df.shape[1] - 1 and pas:
                        stable = True
                        print(f"Stable after {numdiff} differencing")
                        numdiff -=1
                        break
                numdiff += 1
            if numdiff > 2:
                print(f"Too many differencing {numdiff}, using 2 instead")
                numdiff = 2
                train_val_df = np.diff(train_val_df_copy, n = numdiff, axis = 0)
            # del train_val_df_copy
            # #--------uncomment to see validation and testing kpss test with train searched numdiff-------
            # diff_val = np.diff(val_df, n = numdiff, axis = 0)
            # diff_test = np.diff(test_df, n = numdiff, axis = 0)
            # print("-----------------validation -------------------")
            # for i in range(diff_val.shape[1]):
            #     with warnings.catch_warnings():
            #         warnings.simplefilter("ignore")
            #         kpss_stat = kpss(diff_val[:,i])
            #     if kpss_stat[1] < 0.05:
            #         print(f"KPSS test failed for {i}th column")

            # print("-----------------testing -------------------")
            # for i in range(diff_test.shape[1]):
            #     with warnings.catch_warnings():
            #         warnings.simplefilter("ignore")
            #         kpss_stat = kpss(diff_test[:,i])
            #     if kpss_stat[1] < 0.05:
            #         print(f"KPSS test failed for {i}th column")
            # print("-----------------------------------------------------")
        # train_df, val_df, test_df = pd.DataFrame(train_df), pd.DataFrame(val_df), pd.DataFrame(test_df)
        elif args.adf:
            from statsmodels.tsa.stattools import adfuller
            numdiff = 0
            stable = False
            train_val_df_copy = copy.deepcopy(train_val_df)
            print("-----------------ADF on training -------------------")
            while stable == False:
                print(f"Trying {numdiff} differencing")
                train_val_df = np.diff(train_val_df_copy, n = numdiff, axis = 0)
                pas = True
                for i in range(train_val_df.shape[1]):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        adf_stat = adfuller(train_val_df[:,i], regression='c', maxlag=510)
                    if adf_stat[1] > 0.02:
                        print(f"ADF test failed for {numdiff} differencing on {i}th column")
                        pas = False
                    if i == train_val_df.shape[1] - 1 and pas:
                        stable = True
                        print(f"Stable after {numdiff} differencing")
                        numdiff -=1
                        break
                numdiff += 1
            if numdiff > 2:
                print(f"Too many differencing {numdiff}, using 2 instead")
                numdiff = 2
                train_val_df = np.diff(train_val_df_copy, n = numdiff, axis = 0)
        else:
            numdiff = 1
            train_val_df_copy = copy.deepcopy(train_val_df)
            train_val_df = np.diff(train_val_df_copy, n = numdiff, axis = 0)

        test_df = pd.DataFrame(test_df)
        train_val_df = pd.DataFrame(train_val_df)
        train_val_df_copy = pd.DataFrame(train_val_df_copy)

        # diff_val = pd.DataFrame(diff_val)
        if args.dataset[:3] == "ILI":
                search_space = {"window_len": 96-numdiff-np.array([0]+[int(1.5**i) for i in range(12)][1:])}
                first_lag = 95
        else:
            search_space = {"window_len": 512-numdiff-np.array([0]+[int(2**i) for i in range(5)][1:])}
            # search_space = {"window_len": 512-numdiff-np.array([0]+[int(1.5**i) for i in range(16)][1:])}

            first_lag = 511

        torch.cuda.empty_cache()
        print("doing differencing:", numdiff)    
        args.do_diff = True if numdiff > 0 else False
        start_time = time.perf_counter()
        ar = AR_diff(input_length, horizon)
        val_mse = ar.fit_raw(train_val_df, train_val_df_copy, DEVICE, args.time_limit_hours, args.do_diff, use_ols=args.use_ols, first_lag=first_lag, search_space=search_space, new_metric=True, num_diff=numdiff)
        print(f"AR search for {prefix} {args.horizon} finished in {time.perf_counter() - start_time} seconds with validation MSE {val_mse}")
        ar.fit_preset(train_val_df, ar.best_lags, args.do_diff, torch.device("cpu"), use_ols=args.use_ols, numdiff=numdiff)
        mse, mae, _, _ = ar.test_loss_acc_df(test_df, torch.device("cpu"), return_prediction=True, numdiff=numdiff)
        print(f"Final AR metrics for {prefix} {args.horizon}: MSE {np.mean(mse)} | MAE {np.mean(mae)}")
        # other_bits = ((time.perf_counter() - start_time)/60, f"Reached Time Limit? {ar.time_is_up}", f"Best Lags {ar.lags}")
    else:
        if args.kpss:
            from statsmodels.tsa.stattools import kpss
            numdiff = 0
            stable = False
            train_df_copy = copy.deepcopy(train_df)
            print("-----------------KPSS on training -------------------")
            while stable == False:
                print(f"Trying {numdiff} differencing")
                train_df = np.diff(train_df_copy, n = numdiff, axis = 0)
                for i in range(train_df.shape[1]):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        kpss_stat = kpss(train_df[:,i])
                    if kpss_stat[1] < 0.05:
                        print(f"KPSS test failed for {numdiff} differencing on {i}th column")
                        numdiff += 1
                        break
                    if i == train_df.shape[1] - 1:
                        stable = True
                        print(f"Stable after {numdiff} differencing")
            if numdiff > 2:
                print("Too many differencing, using 2 instead")
                numdiff = 2
                train_df = np.diff(train_df_copy, n = numdiff, axis = 0)
            
            #--------uncomment to see validation and testing kpss test with train searched numdiff-------
            # diff_val = np.diff(val_df, n = numdiff, axis = 0)
            # diff_test = np.diff(test_df, n = numdiff, axis = 0)
            # print("-----------------validation -------------------")
            # for i in range(diff_val.shape[1]):
            #     with warnings.catch_warnings():
            #         warnings.simplefilter("ignore")
            #         kpss_stat = kpss(diff_val[:,i])
            #     if kpss_stat[1] < 0.05:
            #         print(f"KPSS test failed for {i}th column")

            # print("-----------------testing -------------------")
            # for i in range(diff_test.shape[1]):
            #     with warnings.catch_warnings():
            #         warnings.simplefilter("ignore")
            #         kpss_stat = kpss(diff_test[:,i])
            #     if kpss_stat[1] < 0.05:
            #         print(f"KPSS test failed for {i}th column")
            # print("-----------------------------------------------------")
        train_df, val_df, test_df = pd.DataFrame(train_df), pd.DataFrame(val_df), pd.DataFrame(test_df)
        train_df_copy = pd.DataFrame(train_df_copy)

        if args.dataset[:3] == "ILI":
                search_space = {"window_len": 96-numdiff-np.array([0]+[int(1.5**i) for i in range(12)][1:])}
                first_lag = 96-numdiff
        else:
            search_space = {"window_len": 512-numdiff-np.array([0]+[int(1.5**i) for i in range(16)][1:])}
            first_lag = 512-numdiff
        search_space = None
        torch.cuda.empty_cache()
        print("doing differencing:", numdiff)  
        args.do_diff = True if numdiff > 0 else False
        start_time = time.perf_counter()
        ar = AR_diff(input_length, horizon)
        val_mse = ar.fit_raw(train_df, val_df, DEVICE, args.time_limit_hours, args.do_diff, use_ols=args.use_ols, first_lag=first_lag, search_space=search_space, num_diff=numdiff)
        print(f"AR search for {prefix} {args.horizon} finished in {time.perf_counter() - start_time} seconds with validation MSE {val_mse}")
        train_val_df_diff = pd.DataFrame(np.diff(pd.concat((train_df, val_df)).values, n = numdiff, axis = 0))
        ar.fit_preset(train_val_df_diff, ar.best_lags, args.do_diff, torch.device("cpu"), use_ols=args.use_ols, numdiff=numdiff)
        # ar.fit_preset(pd.concat((train_df_copy, val_df)), ar.best_lags, args.do_diff, torch.device("cpu"), use_ols=args.use_ols, numdiff=numdiff)
        # val_mse, val_mae = ar.test_loss_acc_df(val_df, torch.device("cpu"),numdiff=numdiff)
        mse, mae = ar.test_loss_acc_df(test_df, torch.device("cpu"),numdiff=numdiff)
        print(f"Final AR metrics for {prefix} {args.horizon}: MSE {np.mean(mse)} | MAE {np.mean(mae)}")
        # print(f"Final AR metrics for {prefix} {args.horizon}: MSE {np.mean(val_mse)} | MAE {np.mean(val_mae)}")


if __name__ == "__main__":
    main()