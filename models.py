import torch
import random
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
import numpy as np 
import time
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from typing import List
import copy

class TSForecaster:
    def __init__(self, input_length, output_length) -> None:
        self.input_length = input_length
        self.output_length = output_length

    def fit_loader(self, train_loader, val_loader, lr, max_epochs):
        raise NotImplementedError()

    def fit_raw(self, scaled_train_df, scaled_val_df):
        raise NotImplementedError()

    def predict(self, input) -> torch.Tensor:
        raise NotImplementedError()
    
    def test_loss_acc(self, test_ds, skip_prop = 0.0, skip_seed = 43):
        mses = list()
        maes = list()
        random.seed(skip_seed)
        for input, truth in tqdm(test_ds):
            if random.random() < skip_prop:
                mses.append(np.nan)
                maes.append(np.nan)
                continue
            preds = self.predict(input)
            mses.append(((preds - truth)**2).mean())
            maes.append(torch.abs(preds - truth).mean())

        return mses, maes
    
    def test_loss_acc_loader(self, test_loader, device, skip_prop = 0.0, skip_seed = 43, progress_bar=True):
        mses = list()
        maes = list()
        random.seed(skip_seed)
        with torch.no_grad():
            for input, truth in (tqdm(test_loader) if progress_bar else test_loader):
                if random.random() < skip_prop:
                    mses.extend([np.nan] * input.shape[0])
                    maes.extend([np.nan] * input.shape[0])
                    continue
                truth = truth.to(device).view(truth.shape[0], -1)
                preds = self.predict(input.to(device))
                preds = preds.view(-1, self.output_length)
                mses.extend(((preds - truth)**2).mean(dim=-1).tolist())
                maes.extend(torch.abs(preds - truth).mean(dim=-1).tolist())

        return mses, maes
    
    def plot_predictions(self, ts_input, truth):
        with torch.no_grad():
            preds = self.predict(ts_input).cpu()
            ts_input = ts_input.cpu()
            truth = truth.cpu()
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(ts_input.flatten(), label="input")
            ax.plot(range(self.input_length + 1, self.input_length + self.output_length + 1), truth.flatten(), label="truth", alpha=0.5)
            ax.plot(range(self.input_length + 1, self.input_length + self.output_length + 1), preds.flatten(), label="forecast")
            ax.legend()

class GlobalMean(TSForecaster):

    def fit_raw(self, scaled_train_df, scaled_val_df):
        all_df = pd.concat([scaled_train_df, scaled_val_df])
        self.mean = all_df.to_numpy().mean()

    def predict(self, input):
        return torch.tensor([self.mean] * self.output_length)



class MovingAverage(TSForecaster):
    def fit_raw(self, scaled_train_df, scaled_val_df):
        all_df = pd.concat([scaled_train_df, scaled_val_df])
        pred_df = all_df.copy()
        self.mses = list()
        for window_length in tqdm(range(1, self.input_length + 1)):
            total_mse = 0
            for output_step in range(1, self.output_length + 1):
                if output_step == 1:
                    pred_df = all_df.rolling(window=window_length).mean().shift(output_step)
                else:
                    pred_df = (((self.input_length + 1) * pred_df - all_df.shift(self.input_length))/self.input_length).shift(1)

                mse = np.nanmean(((
                    all_df.iloc[self.input_length + output_step:-(self.output_length - output_step) or None] - 
                    pred_df[self.input_length + output_step:-(self.output_length - output_step) or None]
                )**2).to_numpy())
                total_mse += mse
            total_mse /= self.output_length
            self.mses.append(total_mse)

        self.window_size = np.argmin(self.mses) + 1
        print(f"Selected Window Size: {self.window_size}")

    def predict(self, input) -> torch.Tensor:
        preds = list()
        for output_step in range(1, self.output_length + 1):
            window_sum = sum(preds[-self.window_size:]) + input.flatten()[-(self.window_size - output_step + 1):0 if output_step > self.window_size else None].sum()
            preds.append(window_sum/self.window_size)
        return torch.tensor(preds)
    
@dataclass 
class DummyAR:
    params: List[float]
    ar_lags: List[float]

def _unfold_df(df, device, input_length, output_length, for_var=False):
    data = torch.tensor(df.to_numpy() if for_var else np.transpose(df.to_numpy()), dtype=torch.float64 if for_var else torch.float64, device=device)
    return data.unfold(0 if for_var else 1, input_length + output_length, 1)

#Implementation of AR model that supports differencing of 0 or 1
class AR_normal(TSForecaster):
    def fit_raw(self, scaled_train_df, scaled_val_df, device, time_limit_hours, do_first_diff: bool, use_ols=False, first_lag=0, last_lag=None, search_space=None):
        #find best lags based on validation set
        print(f"{time_limit_hours} hours limit")

        self.do_first_diff = do_first_diff

        train_series = np.transpose(scaled_train_df.to_numpy()).flatten()
        if self.do_first_diff:
            train_series = np.diff(train_series, 1, axis=0)
        val_series_rw = _unfold_df(scaled_val_df, device, self.input_length, self.output_length)
        self.mses = list()
        start_time = time.perf_counter()
        max_lags = self.input_length if not do_first_diff else self.input_length - 1
        if last_lag is not None:
            max_lags = last_lag
        self.time_is_up = False
        space = range(first_lag, max_lags + 1, 1) if search_space is None else np.flip(search_space["window_len"],0).tolist()
        for lags in tqdm(space):
            if use_ols:
                if do_first_diff:
                    betas = self._fit_ols(scaled_train_df.diff(1).iloc[1:, :], device, lags)
                else:
                    betas = self._fit_ols(scaled_train_df, device, lags)
                model = DummyAR(
                    params=betas,
                    ar_lags=None if lags == 0 else betas[1:]
                )
            else:
                model = AutoReg(
                    endog=train_series,
                    lags=lags,
                    trend='c'
                ).fit()

            val_mse, _ = self._test_loss_acc_stack(model, val_series_rw, device)
            self.mses.append(val_mse)
            if (time.perf_counter() - start_time)/3600 > time_limit_hours:
                print(f"{time_limit_hours} hours have passed, time's up")
                self.time_is_up = True
                break
        print(f"Best Lags: {space[np.argmin(self.mses)]}")
        self.best_lags = space[np.argmin(self.mses)]
        return np.min(self.mses)

    def _fit_ols(self, scaled_train_df, device, input_length):
        #fit the linear regression model using OLS
        torch.cuda.empty_cache()
        data_tensor = _unfold_df(scaled_train_df, torch.device("cpu"), input_length=input_length, output_length=1)
        y = copy.deepcopy(data_tensor[:, :, input_length].reshape(-1))
        y.to(device)
        if input_length == 0:
            del data_tensor
            betas = y.mean()[None]
        else:
            x = torch.nn.functional.pad(copy.deepcopy(data_tensor[:, :, :input_length]).reshape(-1, input_length), (0, 1), value=1)
            del data_tensor
            x.to(device)
            betas = torch.inverse(torch.matmul(torch.transpose(x, 1, 0), x))
            betas = torch.matmul(betas, torch.transpose(x, 1, 0))
            betas = torch.matmul(betas, y)
            del x, y
        return torch.flip(betas, [0])

    def fit_preset(self, scaled_train_df, lags, do_first_diff: bool, device: torch.device, use_ols=False):
        #fit the model with the best lags
        self.do_first_diff = do_first_diff
        self.lags = lags
        if use_ols:
            if do_first_diff:
                betas = self._fit_ols(scaled_train_df.diff(1).iloc[1:, :], device, lags)
            else:
                betas = self._fit_ols(scaled_train_df, device, lags)
            self.model = DummyAR(
                params=betas,
                ar_lags=None if lags == 0 else betas[1:]
            )
        else:
            train_series = np.transpose(scaled_train_df.to_numpy()).flatten()
            if self.do_first_diff:
                train_series = np.diff(train_series, 1, axis=0)
            self.model = AutoReg(
                    endog=train_series,
                    lags=self.lags,
                    trend='c'
            ).fit()
        print(f"Selected Lags: {self.lags}")
    
    def _predict_stack(self, model, val_series_rw, device: torch.device) -> torch.Tensor:
        constant = model.params[0]

        if not self.do_first_diff:
            if model.ar_lags is None:
                return constant
        
            lags = len(model.ar_lags)
            lag_params = torch.flip(torch.tensor(model.params[1:], dtype=torch.float32, device=device), [0])
            assert len(lag_params) == lags

            val_preds = torch.zeros(val_series_rw.shape[0], val_series_rw.shape[1], self.output_length, dtype=torch.float32, device=device)
            for t in range(self.output_length):
                if t == 0:
                    val_preds[:, :, t] = (val_series_rw[:, :, -lags - self.output_length:-self.output_length] * lag_params).sum(dim=-1) 
                elif t < lags:
                    val_preds[:, :, t] = (val_series_rw[:, :, -lags + t - self.output_length:-self.output_length] * lag_params[:-t]).sum(dim=-1) + (val_preds[:, :, :t] * lag_params[-t:]).sum(dim=-1)
                else:
                    val_preds[:, :, t] = (val_preds[:, :, t-lags:t] * lag_params).sum(dim=-1)
                val_preds[:, :, t] = val_preds[:, :, t] + constant 
            return val_preds
        else:
            if model.ar_lags is None:
                deltas = torch.arange(1, self.output_length + 1, 1, device=device) * constant
                val_preds = val_series_rw[:, :, self.input_length - 1][:, :, None] + deltas
                return val_preds
            
            lags = len(model.ar_lags)
            lag_params = torch.flip(torch.tensor(model.params[1:], dtype=torch.float32, device=device), [0])
            assert len(lag_params) == lags

            val_diffs = torch.diff(val_series_rw, 1, dim=-1)
            val_diff_preds = torch.zeros(val_series_rw.shape[0], val_series_rw.shape[1], self.output_length, dtype=torch.float32, device=device)
            for t in range(self.output_length):
                if t == 0:
                    val_diff_preds[:, :, t] = (val_diffs[:, :, -lags - self.output_length:-self.output_length] * lag_params).sum(dim=-1)
                elif t < lags:
                    val_diff_preds[:, :, t] = (val_diffs[:, :, -lags + t - self.output_length:-self.output_length] * lag_params[:-t]).sum(dim=-1) + (val_diff_preds[:, :, :t] * lag_params[-t:]).sum(dim=-1)
                else:
                    val_diff_preds[:, :, t] = (val_diff_preds[:, :, t-lags:t] * lag_params).sum(dim=-1)
                val_diff_preds[:, :, t] = val_diff_preds[:, :, t] + constant 
            val_preds = val_series_rw[:, :, self.input_length - 1, None] + torch.cumsum(val_diff_preds, dim=-1)
            return val_preds

    def _test_loss_acc_stack(self, model, val_series_rw, device: torch.device, return_prediction=False):
        val_preds = self._predict_stack(model, val_series_rw, device)
        if return_prediction:
            return (
                ((val_preds - val_series_rw[:, :, -self.output_length:])**2).mean().item(), 
                torch.abs(val_preds - val_series_rw[:, :, -self.output_length:]).mean().item(),
                val_preds,
                val_series_rw[:, :, -self.output_length:]
            )
        return (
            ((val_preds - val_series_rw[:, :, -self.output_length:])**2).mean().item(), 
            torch.abs(val_preds - val_series_rw[:, :, -self.output_length:]).mean().item()
        )
       
    
    def test_loss_acc_df(self, scaled_test_df, device: torch.device, return_prediction=False):
        test_series_rw = _unfold_df(scaled_test_df, device, self.input_length, self.output_length)
        # print(test_series_rw.shape)
        return self._test_loss_acc_stack(self.model, test_series_rw, device, return_prediction)

    def predict(self, input) -> torch.Tensor:
        preds = self.model.append(input.flatten(), refit=False).forecast(steps=self.output_length)
        return torch.tensor(preds)
    
#Implementation of AR model that supports differencing from 0 to 2.
class AR_diff(TSForecaster):
    def fit_raw(self, scaled_train_df, scaled_val_df, device, time_limit_hours, do_first_diff: bool, num_diff: int, use_ols=False, first_lag=0, last_lag=None, search_space=None, new_metric = False):
        #find best lags based on validation set
        print(f"{time_limit_hours} hours limit")
        self.numdiff = num_diff
        self.do_first_diff = do_first_diff
        val_series_rw = _unfold_df(scaled_val_df, device, self.input_length, self.output_length)
        self.mses = list()
        start_time = time.perf_counter()
        if self.numdiff == 0:
            max_lags = self.input_length
        elif self.numdiff == 1:
            max_lags = self.input_length - 1
        elif self.numdiff == 2:
            max_lags = self.input_length - 2
        else:
            print("Invalid num_diff")

        if last_lag is not None:
            max_lags = last_lag
        self.time_is_up = False
        space = range(first_lag, max_lags + 1, 1) if search_space is None else np.flip(search_space["window_len"],0).tolist()
        for lags in tqdm(space):
            if use_ols:
                betas = self._fit_ols(scaled_train_df, device, lags)
                model = DummyAR(
                    params=betas,
                    ar_lags=None if lags == 0 else betas[1:]
                )
            else:
                train_series = np.transpose(scaled_train_df.to_numpy()).flatten()
                model = AutoReg(
                    endog=train_series,
                    lags=lags,
                    trend='c'
                ).fit()

            
            if new_metric:
                val_mse, _, preds, _ = self._test_loss_acc_stack(model, val_series_rw, device, return_prediction=True)
                total_pred = preds.shape[0]*preds.shape[1]*preds.shape[2]
                del preds
                bic = np.log(val_mse) + (lags+1)*np.log(total_pred)/total_pred
                aic = np.log(val_mse) + 2*(lags+1)/total_pred
                score = bic
            else:
                val_mse, _ = self._test_loss_acc_stack(model, val_series_rw, device)
                score = val_mse
            self.mses.append(score)
            if (time.perf_counter() - start_time)/3600 > time_limit_hours:
                print(f"{time_limit_hours} hours have passed, time's up")
                self.time_is_up = True
                break
        print(f"Best Lags: {space[np.argmin(self.mses)]}")
        self.best_lags = space[np.argmin(self.mses)]
        return np.min(self.mses)

    def _fit_ols(self, scaled_train_df, device, input_length):
        #fit the linear regression model using OLS
        torch.cuda.empty_cache()
        data_tensor = _unfold_df(scaled_train_df, torch.device("cpu"), input_length=input_length, output_length=1)
        y = copy.deepcopy(data_tensor[:, :, input_length].reshape(-1))
        y.to(device)
        if input_length == 0:
            del data_tensor
            betas = y.mean()[None]
        else:
            x = torch.nn.functional.pad(copy.deepcopy(data_tensor[:, :, :input_length]).reshape(-1, input_length), (0, 1), value=1)
            del data_tensor
            x.to(device)
            betas = torch.inverse(torch.matmul(torch.transpose(x, 1, 0), x))
            betas = torch.matmul(betas, torch.transpose(x, 1, 0))
            betas = torch.matmul(betas, y)
            del x, y
        return torch.flip(betas, [0])

    # def _unfold_df(self, df, device, input_length=None, output_length=None):
    #     data = torch.tensor(np.transpose(df.to_numpy()), dtype=torch.float32, device=device)
    #     if input_length is None:
    #         input_length = self.input_length
    #     if output_length is None:
    #         output_length = self.output_length
    #     return data.unfold(1, input_length + output_length, 1)

    def fit_preset(self, scaled_train_df, lags, do_first_diff: bool, device: torch.device, numdiff:int, use_ols=False):
        #fit the model with the best lags
        self.do_first_diff = do_first_diff
        self.numdiff = numdiff
        self.lags = lags
        if use_ols:
            betas = self._fit_ols(scaled_train_df, device, lags)
            # if self.numdiff == 1:
            #     betas = self._fit_ols(scaled_train_df.diff(1).iloc[1:, :], device, lags)
            # elif self.numdiff == 2:
            #     betas = self._fit_ols(scaled_train_df.diff(2).iloc[2:, :], device, lags)
            # else:
            #     betas = self._fit_ols(scaled_train_df, device, lags)
            self.model = DummyAR(
                params=betas,
                ar_lags=None if lags == 0 else betas[1:]
            )
        else:
            train_series = np.transpose(scaled_train_df.to_numpy()).flatten()
            if self.do_first_diff:
                train_series = np.diff(train_series, 1, axis=0)
            self.model = AutoReg(
                    endog=train_series,
                    lags=self.lags,
                    trend='c'
            ).fit()
        print(f"Selected Lags: {self.lags}")
    
    def _predict_stack(self, model, val_series_rw, device: torch.device) -> torch.Tensor:
        constant = model.params[0]

        if not self.do_first_diff:
            if model.ar_lags is None:
                return constant
        
            lags = len(model.ar_lags)
            lag_params = torch.flip(torch.tensor(model.params[1:], dtype=torch.float32, device=device), [0])
            assert len(lag_params) == lags

            val_preds = torch.zeros(val_series_rw.shape[0], val_series_rw.shape[1], self.output_length, dtype=torch.float32, device=device)
            for t in range(self.output_length):
                if t == 0:
                    val_preds[:, :, t] = (val_series_rw[:, :, -lags - self.output_length:-self.output_length] * lag_params).sum(dim=-1) 
                elif t < lags:
                    val_preds[:, :, t] = (val_series_rw[:, :, -lags + t - self.output_length:-self.output_length] * lag_params[:-t]).sum(dim=-1) + (val_preds[:, :, :t] * lag_params[-t:]).sum(dim=-1)
                else:
                    val_preds[:, :, t] = (val_preds[:, :, t-lags:t] * lag_params).sum(dim=-1)
                val_preds[:, :, t] = val_preds[:, :, t] + constant 
            return val_preds
        else:
            if self.numdiff == 1:
                if model.ar_lags is None:
                    deltas = torch.arange(1, self.output_length + 1, 1, device=device) * constant
                    val_preds = val_series_rw[:, :, self.input_length - 1][:, :, None] + deltas
                    return val_preds
                
                lags = len(model.ar_lags)
                lag_params = torch.flip(torch.tensor(model.params[1:], dtype=torch.float32, device=device), [0])
                assert len(lag_params) == lags

                val_diffs = torch.diff(val_series_rw, 1, dim=-1)
                val_diff_preds = torch.zeros(val_series_rw.shape[0], val_series_rw.shape[1], self.output_length, dtype=torch.float32, device=device)
                for t in range(self.output_length):
                    if t == 0:
                        val_diff_preds[:, :, t] = (val_diffs[:, :, -lags - self.output_length:-self.output_length] * lag_params).sum(dim=-1)
                    elif t < lags:
                        val_diff_preds[:, :, t] = (val_diffs[:, :, -lags + t - self.output_length:-self.output_length] * lag_params[:-t]).sum(dim=-1) + (val_diff_preds[:, :, :t] * lag_params[-t:]).sum(dim=-1)
                    else:
                        val_diff_preds[:, :, t] = (val_diff_preds[:, :, t-lags:t] * lag_params).sum(dim=-1)
                    val_diff_preds[:, :, t] = val_diff_preds[:, :, t] + constant 
                val_preds = val_series_rw[:, :, self.input_length - 1, None] + torch.cumsum(val_diff_preds, dim=-1)
                return val_preds
            elif self.numdiff == 2:
                
                lags = len(model.ar_lags)
                lag_params = torch.flip(torch.tensor(model.params[1:], dtype=torch.float32, device=device), [0])
                assert len(lag_params) == lags

                val_diffs = torch.diff(val_series_rw, 2, dim=-1)
                val_diff_preds = torch.zeros(val_series_rw.shape[0], val_series_rw.shape[1], self.output_length, dtype=torch.float32, device=device)
                for t in range(self.output_length):
                    if t == 0:
                        val_diff_preds[:, :, t] = (val_diffs[:, :, -lags - self.output_length:-self.output_length] * lag_params).sum(dim=-1)
                    elif t < lags:
                        val_diff_preds[:, :, t] = (val_diffs[:, :, -lags + t - self.output_length:-self.output_length] * lag_params[:-t]).sum(dim=-1) + (val_diff_preds[:, :, :t] * lag_params[-t:]).sum(dim=-1)
                    else:
                        val_diff_preds[:, :, t] = (val_diff_preds[:, :, t-lags:t] * lag_params).sum(dim=-1)
                    val_diff_preds[:, :, t] = val_diff_preds[:, :, t] + constant 
                val_first_diff = val_series_rw[:, :, self.input_length - 1, None] - val_series_rw[:, :, self.input_length - 2, None]+ torch.cumsum(val_diff_preds, dim=-1)
                val_preds = val_series_rw[:, :, self.input_length - 1, None] + torch.cumsum(val_first_diff, dim=-1)
                # print(val_preds.shape)
                return val_preds


    def _test_loss_acc_stack(self, model, val_series_rw, device: torch.device, return_prediction=False):
        val_preds = self._predict_stack(model, val_series_rw, device)
        # if return_prediction:
        #     print(val_preds.shape)
        #     print(val_series_rw[:, :, -self.output_length:].shape)
        if return_prediction:
            return (
                ((val_preds - val_series_rw[:, :, -self.output_length:])**2).mean().item(), 
                torch.abs(val_preds - val_series_rw[:, :, -self.output_length:]).mean().item(),
                val_preds,
                val_series_rw[:, :, -self.output_length:]
            )
        return (
            ((val_preds - val_series_rw[:, :, -self.output_length:])**2).mean().item(), 
            torch.abs(val_preds - val_series_rw[:, :, -self.output_length:]).mean().item()
        )
       
    
    def test_loss_acc_df(self, scaled_test_df, device: torch.device, numdiff: int, return_prediction=False):
        self.numdiff = numdiff
        test_series_rw = _unfold_df(scaled_test_df, device, self.input_length, self.output_length)
        # print(test_series_rw.shape)
        return self._test_loss_acc_stack(self.model, test_series_rw, device, return_prediction)

    def predict(self, input) -> torch.Tensor:
        preds = self.model.append(input.flatten(), refit=False).forecast(steps=self.output_length)
        return torch.tensor(preds)
    

    
class LinearOLS(TSForecaster):
    def __init__(self, input_length, output_length, device: torch.device) -> None:
        super().__init__(input_length, output_length)
        self.device = device

    def _fit_ols(self, scaled_train_df, input_length, device=None):
        torch.cuda.empty_cache()
        data_tensor = _unfold_df(scaled_train_df, device or self.device, input_length, self.output_length)
        y = data_tensor[:, :, input_length:input_length+self.output_length].reshape(-1, self.output_length)
        y.to(device)
        if input_length == 0:
            del data_tensor
            betas = y.mean(dim=0)[None]
        else:
            x = torch.nn.functional.pad(data_tensor[:, :, :input_length].reshape(-1, input_length), (0, 1), value=1)
            del data_tensor
            x.to(device)
            betas = torch.inverse(torch.matmul(torch.transpose(x, 1, 0), x))
            betas = torch.matmul(betas, torch.transpose(x, 1, 0))
            betas = torch.matmul(betas, y)
        return betas
    
    def fit_raw(self, scaled_train_df, scaled_val_df, time_limit_hours, min_input_length=0, max_input_length=None):
        max_input_length = max_input_length or self.input_length
        val_series_rw = _unfold_df(scaled_val_df, self.device, self.input_length, self.output_length)
        self.mses = list()
        start_time = time.perf_counter()
        for lags in tqdm(range(min_input_length, max_input_length + 1, 1), desc="Searching Lags"):
            self.betas = self._fit_ols(scaled_train_df, lags, self.device)
            val_mse, _ = self._test_loss_acc_stack(val_series_rw, lags, 1)
            self.mses.append(val_mse)
            if (time.perf_counter() - start_time)/3600 > time_limit_hours:
                print(f"{time_limit_hours} hours have passed, time's up")
                self.time_is_up = True
                break
        
        print(f"Best Lags: {np.argmin(self.mses) + min_input_length}")
        self.best_lags = np.argmin(self.mses) + min_input_length

    
    def fit_preset(self, scaled_train_df, input_length, time_limit_hours, device=None):
        self.betas = self._fit_ols(scaled_train_df, input_length, device or self.device)
        self.best_lags = input_length
    
    def _predict_stack(self, val_predictors_rw, shrinkage) -> torch.Tensor:
        constant = self.betas[-1] * shrinkage
        if self.betas.shape[0] == 1:
            return constant.expand(val_predictors_rw.shape[-2], -1)
        lag_params = self.betas[:-1] * shrinkage
    
        return torch.matmul(val_predictors_rw, lag_params) + constant
    
    def _test_loss_acc_stack(self, val_series_rw, input_length, shrinkage) -> float:
        val_preds = self._predict_stack(val_series_rw[:, :, -input_length-self.output_length:-self.output_length], shrinkage)
        return (
            ((val_preds - val_series_rw[:, :, -self.output_length:])**2).mean().item(), 
            torch.abs(val_preds - val_series_rw[:, :, -self.output_length:]).mean().item()
        )
    
    def test_loss_acc_df(self, scaled_test_df, shrinkage: float=1, device: torch.device = None):
        test_series_rw = _unfold_df(scaled_test_df, device or self.device, self.best_lags, self.output_length)
        return self._test_loss_acc_stack(test_series_rw, self.best_lags, shrinkage)
    


class AR_zero(TSForecaster):
    def fit_raw(self, scaled_train_df, scaled_val_df, device, time_limit_hours, do_first_diff: bool, num_diff: int, use_ols=False, first_lag=0, last_lag=None, search_space=None, new_metric = False):
        #find best lags based on validation set
        print(f"{time_limit_hours} hours limit")
        self.numdiff = num_diff
        self.do_first_diff = do_first_diff
        val_series_rw = _unfold_df(scaled_val_df, device, self.input_length, self.output_length)
        self.mses = list()
        start_time = time.perf_counter()
        if self.numdiff == 0:
            max_lags = self.input_length
        elif self.numdiff == 1:
            max_lags = self.input_length - 1
        elif self.numdiff == 2:
            max_lags = self.input_length - 2
        else:
            print("Invalid num_diff")

        if last_lag is not None:
            max_lags = last_lag
        self.time_is_up = False
        space = range(first_lag, max_lags + 1, 1) if search_space is None else np.flip(search_space["window_len"],0).tolist()
        mset = 0
        maet = 0
        lagst = 0
        T = len(scaled_train_df) - self.input_length - self.output_length
        for t in range(T):
            all_mse = []
            all_mae = []
            all_score = []
            for lags in space:
                new_scaled_train = scaled_train_df[t:t+self.input_length]
                new_val_series_rw = scaled_val_df[t:t+self.input_length+self.output_length]
                if self.do_first_diff:
                    betas = self._fit_ols(new_scaled_train.diff(num_diff).iloc[num_diff:, :], device, lags)
                else:
                    betas = self._fit_ols(new_scaled_train, device, lags)
                model = DummyAR(
                    params=betas,
                    ar_lags=None if lags == 0 else betas[1:]
                )
                # if new_metric:
                #     # val_mse, _, preds, _ = self._test_loss_acc_stack(model, val_series_rw, device, return_prediction=True)
                #     val_mse, _, preds, _ = self._test_loss_acc_stack(model, val_series_rw, torch.device("cpu"), return_prediction=True)
                #     total_pred = preds.shape[0]*preds.shape[1]*preds.shape[2]
                #     del preds
                #     bic = np.log(val_mse) + (lags+1)*np.log(total_pred)/total_pred
                #     aic = np.log(val_mse) + 2*(lags+1)/total_pred
                #     score = bic
                # else:
                #     val_mse, _ = self._test_loss_acc_stack(model, val_series_rw, device)
                #     score = val_mse
                val_mse, val_mae, preds, _ = self.test_loss_acc_df(new_val_series_rw, device, numdiff=num_diff, return_prediction=True, model=model)
                num = preds.shape[0]*preds.shape[1]*preds.shape[2]
                bic = np.log(val_mse) + (lags+1)*np.log(num)/num
                aic = np.log(val_mse) + 2*(lags+1)/num
                if new_metric:
                    score = bic
                else:
                    score = val_mse
                all_mse.append(val_mse)
                all_mae.append(val_mae)
                all_score.append(score)
            best_score = np.min(all_score)
            best_mse = all_mse[np.argmin(all_score)]
            best_mae = all_mae[np.argmin(all_score)]
            best_lags = space[np.argmin(all_score)]
            mset += best_mse
            maet += best_mae
            lagst += best_lags
            if (time.perf_counter() - start_time)/3600 > time_limit_hours:
                print(f"{time_limit_hours} hours have passed, time's up")
                self.time_is_up = True
                break
        avg_mse = mset/T
        avg_mae = maet/T
        avg_lags = lagst/T
        print(f"Best Lags: {avg_lags}")
        return avg_mse, avg_mae

    def _fit_ols(self, scaled_train_df, device, input_length):
        #fit the linear regression model using OLS
        torch.cuda.empty_cache()
        data_tensor = _unfold_df(scaled_train_df, torch.device("cpu"), input_length=input_length, output_length=1)
        y = copy.deepcopy(data_tensor[:, :, input_length].reshape(-1))
        y.to(device)
        if input_length == 0:
            del data_tensor
            betas = y.mean()[None]
        else:
            x = torch.nn.functional.pad(copy.deepcopy(data_tensor[:, :, :input_length]).reshape(-1, input_length), (0, 1), value=1)
            del data_tensor
            x.to(device)
            betas = torch.inverse(torch.matmul(torch.transpose(x, 1, 0), x))
            betas = torch.matmul(betas, torch.transpose(x, 1, 0))
            betas = torch.matmul(betas, y)
            del x, y
        return torch.flip(betas, [0])
    
    def _predict_stack(self, model, val_series_rw, device: torch.device) -> torch.Tensor:
        constant = model.params[0]

        if not self.do_first_diff:
            if model.ar_lags is None:
                return constant
        
            lags = len(model.ar_lags)
            lag_params = torch.flip(torch.tensor(model.params[1:], dtype=torch.float32, device=device), [0])
            assert len(lag_params) == lags

            val_preds = torch.zeros(val_series_rw.shape[0], val_series_rw.shape[1], self.output_length, dtype=torch.float32, device=device)
            for t in range(self.output_length):
                if t == 0:
                    val_preds[:, :, t] = (val_series_rw[:, :, -lags - self.output_length:-self.output_length] * lag_params).sum(dim=-1) 
                elif t < lags:
                    val_preds[:, :, t] = (val_series_rw[:, :, -lags + t - self.output_length:-self.output_length] * lag_params[:-t]).sum(dim=-1) + (val_preds[:, :, :t] * lag_params[-t:]).sum(dim=-1)
                else:
                    val_preds[:, :, t] = (val_preds[:, :, t-lags:t] * lag_params).sum(dim=-1)
                val_preds[:, :, t] = val_preds[:, :, t] + constant 
            return val_preds
        else:
            if self.numdiff == 1:
                if model.ar_lags is None:
                    deltas = torch.arange(1, self.output_length + 1, 1, device=device) * constant
                    val_preds = val_series_rw[:, :, self.input_length - 1][:, :, None] + deltas
                    return val_preds
                
                lags = len(model.ar_lags)
                lag_params = torch.flip(torch.tensor(model.params[1:], dtype=torch.float32, device=device), [0])
                assert len(lag_params) == lags

                val_diffs = torch.diff(val_series_rw, 1, dim=-1).to(device)
                val_series_rw = val_series_rw.to(device)
                val_diff_preds = torch.zeros(val_series_rw.shape[0], val_series_rw.shape[1], self.output_length, dtype=torch.float32, device=device)
                for t in range(self.output_length):
                    if t == 0:
                        val_diff_preds[:, :, t] = (val_diffs[:, :, -lags - self.output_length:-self.output_length] * lag_params).sum(dim=-1)
                    elif t < lags:
                        val_diff_preds[:, :, t] = (val_diffs[:, :, -lags + t - self.output_length:-self.output_length] * lag_params[:-t]).sum(dim=-1) + (val_diff_preds[:, :, :t] * lag_params[-t:]).sum(dim=-1)
                    else:
                        val_diff_preds[:, :, t] = (val_diff_preds[:, :, t-lags:t] * lag_params).sum(dim=-1)
                    val_diff_preds[:, :, t] = val_diff_preds[:, :, t] + constant 
                val_preds = val_series_rw[:, :, self.input_length - 1, None] + torch.cumsum(val_diff_preds, dim=-1)
                return val_preds
            elif self.numdiff == 2:
                
                lags = len(model.ar_lags)
                lag_params = torch.flip(torch.tensor(model.params[1:], dtype=torch.float32, device=device), [0])
                assert len(lag_params) == lags

                val_diffs = torch.diff(val_series_rw, 2, dim=-1)
                val_diff_preds = torch.zeros(val_series_rw.shape[0], val_series_rw.shape[1], self.output_length, dtype=torch.float32, device=device)
                for t in range(self.output_length):
                    if t == 0:
                        val_diff_preds[:, :, t] = (val_diffs[:, :, -lags - self.output_length:-self.output_length] * lag_params).sum(dim=-1)
                    elif t < lags:
                        val_diff_preds[:, :, t] = (val_diffs[:, :, -lags + t - self.output_length:-self.output_length] * lag_params[:-t]).sum(dim=-1) + (val_diff_preds[:, :, :t] * lag_params[-t:]).sum(dim=-1)
                    else:
                        val_diff_preds[:, :, t] = (val_diff_preds[:, :, t-lags:t] * lag_params).sum(dim=-1)
                    val_diff_preds[:, :, t] = val_diff_preds[:, :, t] + constant 
                val_first_diff = val_series_rw[:, :, self.input_length - 1, None] - val_series_rw[:, :, self.input_length - 2, None]+ torch.cumsum(val_diff_preds, dim=-1)
                val_preds = val_series_rw[:, :, self.input_length - 1, None] + torch.cumsum(val_first_diff, dim=-1)
                # print(val_preds.shape)
                return val_preds


    def _test_loss_acc_stack(self, model, val_series_rw, device: torch.device, return_prediction=False):
        val_series_rw = val_series_rw.to(device)
        val_preds = self._predict_stack(model, val_series_rw, device)
        # if return_prediction:
        #     print(val_preds.shape)
        #     print(val_series_rw[:, :, -self.output_length:].shape)
        if return_prediction:
            return (
                ((val_preds - val_series_rw[:, :, -self.output_length:])**2).mean().item(), 
                torch.abs(val_preds - val_series_rw[:, :, -self.output_length:]).mean().item(),
                val_preds,
                val_series_rw[:, :, -self.output_length:]
            )
        return (
            ((val_preds - val_series_rw[:, :, -self.output_length:])**2).mean().item(), 
            torch.abs(val_preds - val_series_rw[:, :, -self.output_length:]).mean().item()
        )
       
    
    def test_loss_acc_df(self, scaled_test_df, device: torch.device, numdiff: int, return_prediction=False, model = None):
        if not model:
            model = self.model
        self.numdiff = numdiff
        test_series_rw = _unfold_df(scaled_test_df, device, self.input_length, self.output_length)
        # print(test_series_rw.shape)
        return self._test_loss_acc_stack(model, test_series_rw, device, return_prediction)

    def predict(self, input) -> torch.Tensor:
        preds = self.model.append(input.flatten(), refit=False).forecast(steps=self.output_length)
        return torch.tensor(preds)
    


#Implementation of AR model with evaluation by steps to allevaite memory overhead.
class AR_diff_step(TSForecaster):
    def fit_raw(self, scaled_train_df, scaled_val_df, device, time_limit_hours, do_first_diff: bool, num_diff: int, use_ols=False, first_lag=0, last_lag=None, search_space=None, new_metric = False):
        #find best lags based on validation set
        print(f"{time_limit_hours} hours limit")
        self.numdiff = num_diff
        self.do_first_diff = do_first_diff
        val_series_rw = _unfold_df(scaled_val_df, torch.device("cpu"), self.input_length, self.output_length)
        self.mses = list()
        start_time = time.perf_counter()
        if self.numdiff == 0:
            max_lags = self.input_length
        elif self.numdiff == 1:
            max_lags = self.input_length - 1
        elif self.numdiff == 2:
            max_lags = self.input_length - 2
        else:
            print("Invalid num_diff")

        if last_lag is not None:
            max_lags = last_lag
        self.time_is_up = False
        space = range(first_lag, max_lags + 1, 1) if search_space is None else np.flip(search_space["window_len"],0).tolist()
        for lags in tqdm(space):
            if use_ols:
                betas = self._fit_ols(scaled_train_df, device, lags)
                model = DummyAR(
                    params=betas,
                    ar_lags=None if lags == 0 else betas[1:]
                )
            else:
                train_series = np.transpose(scaled_train_df.to_numpy()).flatten()
                model = AutoReg(
                    endog=train_series,
                    lags=lags,
                    trend='c'
                ).fit()

            
            if new_metric:
                # val_mse, _, preds, _ = self._test_loss_acc_stack(model, val_series_rw, device, return_prediction=True)
                val_mse, _, preds, _ = self._test_loss_acc_stack(model, val_series_rw, torch.device("cpu"), return_prediction=True)
                total_pred = preds.shape[1]*preds.shape[2]*preds.shape[0]
                del preds
                bic = np.log(val_mse) + (lags+1)*np.log(total_pred)/total_pred
                aic = np.log(val_mse) + 2*(lags+1)/total_pred
                score = bic
            else:
                val_mse, _ = self._test_loss_acc_stack(model, val_series_rw, device)
                score = val_mse
            self.mses.append(score)
            if (time.perf_counter() - start_time)/3600 > time_limit_hours:
                print(f"{time_limit_hours} hours have passed, time's up")
                self.time_is_up = True
                break
        print(f"Best Lags: {space[np.argmin(self.mses)]}")
        self.best_lags = space[np.argmin(self.mses)]
        return np.min(self.mses)

    def _fit_ols(self, scaled_train_df, device, input_length):
        #fit the linear regression model using OLS
        torch.cuda.empty_cache()
        data_tensor = _unfold_df(scaled_train_df, torch.device("cpu"), input_length=input_length, output_length=1)
        y = copy.deepcopy(data_tensor[:, :, input_length].reshape(-1))
        y.to(device)
        if input_length == 0:
            del data_tensor
            betas = y.mean()[None]
        else:
            x = torch.nn.functional.pad(copy.deepcopy(data_tensor[:, :, :input_length]).reshape(-1, input_length), (0, 1), value=1)
            del data_tensor
            x.to(device)
            betas = torch.inverse(torch.matmul(torch.transpose(x, 1, 0), x))
            betas = torch.matmul(betas, torch.transpose(x, 1, 0))
            betas = torch.matmul(betas, y)
            del x, y
            torch.cuda.empty_cache()
        return torch.flip(betas, [0])

    # def _unfold_df(self, df, device, input_length=None, output_length=None):
    #     data = torch.tensor(np.transpose(df.to_numpy()), dtype=torch.float32, device=device)
    #     if input_length is None:
    #         input_length = self.input_length
    #     if output_length is None:
    #         output_length = self.output_length
    #     return data.unfold(1, input_length + output_length, 1)

    def fit_preset(self, scaled_train_df, lags, do_first_diff: bool, device: torch.device, numdiff:int, use_ols=False):
        #fit the model with the best lags
        self.do_first_diff = do_first_diff
        self.numdiff = numdiff
        self.lags = lags
        if use_ols:
            betas = self._fit_ols(scaled_train_df, device, lags)
            # if self.numdiff == 1:
            #     betas = self._fit_ols(scaled_train_df.diff(1).iloc[1:, :], device, lags)
            # elif self.numdiff == 2:
            #     betas = self._fit_ols(scaled_train_df.diff(2).iloc[2:, :], device, lags)
            # else:
            #     betas = self._fit_ols(scaled_train_df, device, lags)
            self.model = DummyAR(
                params=betas,
                ar_lags=None if lags == 0 else betas[1:]
            )
        else:
            train_series = np.transpose(scaled_train_df.to_numpy()).flatten()
            if self.do_first_diff:
                train_series = np.diff(train_series, 1, axis=0)
            self.model = AutoReg(
                    endog=train_series,
                    lags=self.lags,
                    trend='c'
            ).fit()
        print(f"Selected Lags: {self.lags}")
    
    def _predict_stack(self, model, val_series_rw, device: torch.device) -> torch.Tensor:
        constant = model.params[0]

        if not self.do_first_diff:
            if model.ar_lags is None:
                return constant
        
            lags = len(model.ar_lags)
            lag_params = torch.flip(torch.tensor(model.params[1:], dtype=torch.float32, device=device), [0])
            assert len(lag_params) == lags

            val_preds = torch.zeros(val_series_rw.shape[0], val_series_rw.shape[1], self.output_length, dtype=torch.float32, device=device)
            for t in range(self.output_length):
                if t == 0:
                    val_preds[:, :, t] = (val_series_rw[:, :, -lags - self.output_length:-self.output_length] * lag_params).sum(dim=-1) 
                elif t < lags:
                    val_preds[:, :, t] = (val_series_rw[:, :, -lags + t - self.output_length:-self.output_length] * lag_params[:-t]).sum(dim=-1) + (val_preds[:, :, :t] * lag_params[-t:]).sum(dim=-1)
                else:
                    val_preds[:, :, t] = (val_preds[:, :, t-lags:t] * lag_params).sum(dim=-1)
                val_preds[:, :, t] = val_preds[:, :, t] + constant 
            return val_preds
        else:
            if self.numdiff == 1:
                if model.ar_lags is None:
                    deltas = torch.arange(1, self.output_length + 1, 1, device=device) * constant
                    val_preds = val_series_rw[:, :, self.input_length - 1][:, :, None] + deltas
                    return val_preds
                
                lags = len(model.ar_lags)
                lag_params = torch.flip(torch.tensor(model.params[1:], dtype=torch.float32, device=device), [0])
                assert len(lag_params) == lags

                val_diffs = torch.diff(val_series_rw, 1, dim=-1).to(device)
                val_series_rw = val_series_rw.to(device)
                val_diff_preds = torch.zeros(val_series_rw.shape[0], val_series_rw.shape[1], self.output_length, dtype=torch.float32, device=device)
                for t in range(self.output_length):
                    if t == 0:
                        val_diff_preds[:, :, t] = (val_diffs[:, :, -lags - self.output_length:-self.output_length] * lag_params).sum(dim=-1)
                    elif t < lags:
                        val_diff_preds[:, :, t] = (val_diffs[:, :, -lags + t - self.output_length:-self.output_length] * lag_params[:-t]).sum(dim=-1) + (val_diff_preds[:, :, :t] * lag_params[-t:]).sum(dim=-1)
                    else:
                        val_diff_preds[:, :, t] = (val_diff_preds[:, :, t-lags:t] * lag_params).sum(dim=-1)
                    val_diff_preds[:, :, t] = val_diff_preds[:, :, t] + constant 
                val_preds = val_series_rw[:, :, self.input_length - 1, None] + torch.cumsum(val_diff_preds, dim=-1)
                return val_preds
            elif self.numdiff == 2:
                
                lags = len(model.ar_lags)
                lag_params = torch.flip(torch.tensor(model.params[1:], dtype=torch.float32, device=device), [0])
                assert len(lag_params) == lags

                val_diffs = torch.diff(val_series_rw, 2, dim=-1)
                val_diff_preds = torch.zeros(val_series_rw.shape[0], val_series_rw.shape[1], self.output_length, dtype=torch.float32, device=device)
                for t in range(self.output_length):
                    if t == 0:
                        val_diff_preds[:, :, t] = (val_diffs[:, :, -lags - self.output_length:-self.output_length] * lag_params).sum(dim=-1)
                    elif t < lags:
                        val_diff_preds[:, :, t] = (val_diffs[:, :, -lags + t - self.output_length:-self.output_length] * lag_params[:-t]).sum(dim=-1) + (val_diff_preds[:, :, :t] * lag_params[-t:]).sum(dim=-1)
                    else:
                        val_diff_preds[:, :, t] = (val_diff_preds[:, :, t-lags:t] * lag_params).sum(dim=-1)
                    val_diff_preds[:, :, t] = val_diff_preds[:, :, t] + constant 
                val_first_diff = val_series_rw[:, :, self.input_length - 1, None] - val_series_rw[:, :, self.input_length - 2, None]+ torch.cumsum(val_diff_preds, dim=-1)
                val_preds = val_series_rw[:, :, self.input_length - 1, None] + torch.cumsum(val_first_diff, dim=-1)
                # print(val_preds.shape)
                return val_preds


    def _test_loss_acc_stack(self, model, val_series_rw, device: torch.device, return_prediction=False, split=5):
        all_mse = 0
        all_mae = 0
        val_preds = []
        # print(val_series_rw.shape)
        for i in range(split):
            val_series_rw_split = val_series_rw[:,i*val_series_rw.shape[1]//split:(i+1)*val_series_rw.shape[1]//split,:]
            val_series_rw_split = val_series_rw_split.to(device)
            val_pred = self._predict_stack(model, val_series_rw_split, device).to(torch.device("cpu"))
            # print(val_pred.shape)
            all_mse += ((val_pred - val_series_rw_split[:, :, -self.output_length:])**2).mean().item()
            all_mae += torch.abs(val_pred - val_series_rw_split[:, :, -self.output_length:]).mean().item()
            val_preds.append(val_pred)
        all_mse /= split
        all_mae /= split
        val_preds = torch.cat(val_preds, dim=1)
        # print(val_preds.shape)
        if return_prediction:
            return (
                all_mse, 
                all_mae,
                val_preds,
                val_series_rw[:, :, -self.output_length:]
            )
        return (
            all_mse, 
            all_mae
        )

        # val_preds = self._predict_stack(model, val_series_rw, device)
        # # if return_prediction:
        # #     print(val_preds.shape)
        # #     print(val_series_rw[:, :, -self.output_length:].shape)
        # if return_prediction:
        #     return (
        #         ((val_preds - val_series_rw[:, :, -self.output_length:])**2).mean().item(), 
        #         torch.abs(val_preds - val_series_rw[:, :, -self.output_length:]).mean().item(),
        #         val_preds,
        #         val_series_rw[:, :, -self.output_length:]
        #     )
        # return (
        #     ((val_preds - val_series_rw[:, :, -self.output_length:])**2).mean().item(), 
        #     torch.abs(val_preds - val_series_rw[:, :, -self.output_length:]).mean().item()
        # )
       
    
    def test_loss_acc_df(self, scaled_test_df, device: torch.device, numdiff: int, return_prediction=False):
        self.numdiff = numdiff
        test_series_rw = _unfold_df(scaled_test_df, device, self.input_length, self.output_length)
        # print(test_series_rw.shape)
        return self._test_loss_acc_stack(self.model, test_series_rw, device, return_prediction)

    def predict(self, input) -> torch.Tensor:
        preds = self.model.append(input.flatten(), refit=False).forecast(steps=self.output_length)
        return torch.tensor(preds)
    

class AR_zero_step(TSForecaster):
    def fit_raw(self, scaled_train_df, scaled_val_df, device, time_limit_hours, do_first_diff: bool, num_diff: int, use_ols=False, first_lag=0, last_lag=None, search_space=None, new_metric = False):
        #find best lags based on validation set
        print(f"{time_limit_hours} hours limit")
        self.numdiff = num_diff
        self.do_first_diff = do_first_diff
        val_series_rw = _unfold_df(scaled_val_df, device, self.input_length, self.output_length)
        self.mses = list()
        start_time = time.perf_counter()
        if self.numdiff == 0:
            max_lags = self.input_length
        elif self.numdiff == 1:
            max_lags = self.input_length - 1
        elif self.numdiff == 2:
            max_lags = self.input_length - 2
        else:
            print("Invalid num_diff")

        if last_lag is not None:
            max_lags = last_lag
        self.time_is_up = False
        space = range(first_lag, max_lags + 1, 1) if search_space is None else np.flip(search_space["window_len"],0).tolist()
        mset = 0
        maet = 0
        lagst = 0
        T = len(scaled_train_df) - self.input_length - self.output_length
        for t in range(T):
            all_mse = []
            all_mae = []
            all_score = []
            for lags in space:
                new_scaled_train = scaled_train_df[t:t+self.input_length]
                new_val_series_rw = scaled_val_df[t:t+self.input_length+self.output_length]
                if self.do_first_diff:
                    betas = self._fit_ols(new_scaled_train.diff(num_diff).iloc[num_diff:, :], device, lags)
                else:
                    betas = self._fit_ols(new_scaled_train, device, lags)
                model = DummyAR(
                    params=betas,
                    ar_lags=None if lags == 0 else betas[1:]
                )
                # if new_metric:
                #     # val_mse, _, preds, _ = self._test_loss_acc_stack(model, val_series_rw, device, return_prediction=True)
                #     val_mse, _, preds, _ = self._test_loss_acc_stack(model, val_series_rw, torch.device("cpu"), return_prediction=True)
                #     total_pred = preds.shape[0]*preds.shape[1]*preds.shape[2]
                #     del preds
                #     bic = np.log(val_mse) + (lags+1)*np.log(total_pred)/total_pred
                #     aic = np.log(val_mse) + 2*(lags+1)/total_pred
                #     score = bic
                # else:
                #     val_mse, _ = self._test_loss_acc_stack(model, val_series_rw, device)
                #     score = val_mse
                val_mse, val_mae, preds, _ = self.test_loss_acc_df(new_val_series_rw, device, numdiff=num_diff, return_prediction=True, model=model)
                num = preds.shape[0]*preds.shape[1]*preds.shape[2]
                bic = np.log(val_mse) + (lags+1)*np.log(num)/num
                aic = np.log(val_mse) + 2*(lags+1)/num
                if new_metric:
                    score = bic
                else:
                    score = val_mse
                all_mse.append(val_mse)
                all_mae.append(val_mae)
                all_score.append(score)
            best_score = np.min(all_score)
            best_mse = all_mse[np.argmin(all_score)]
            best_mae = all_mae[np.argmin(all_score)]
            best_lags = space[np.argmin(all_score)]
            mset += best_mse
            maet += best_mae
            lagst += best_lags
            if (time.perf_counter() - start_time)/3600 > time_limit_hours:
                print(f"{time_limit_hours} hours have passed, time's up")
                self.time_is_up = True
                break
        avg_mse = mset/T
        avg_mae = maet/T
        avg_lags = lagst/T
        print(f"Best Lags: {avg_lags}")
        return avg_mse, avg_mae

    def _fit_ols(self, scaled_train_df, device, input_length):
        #fit the linear regression model using OLS
        torch.cuda.empty_cache()
        data_tensor = _unfold_df(scaled_train_df, torch.device("cpu"), input_length=input_length, output_length=1)
        y = copy.deepcopy(data_tensor[:, :, input_length].reshape(-1))
        y.to(device)
        if input_length == 0:
            del data_tensor
            betas = y.mean()[None]
        else:
            x = torch.nn.functional.pad(copy.deepcopy(data_tensor[:, :, :input_length]).reshape(-1, input_length), (0, 1), value=1)
            del data_tensor
            x.to(device)
            betas = torch.inverse(torch.matmul(torch.transpose(x, 1, 0), x))
            betas = torch.matmul(betas, torch.transpose(x, 1, 0))
            betas = torch.matmul(betas, y)
            del x, y
        return torch.flip(betas, [0])
    
    def _predict_stack(self, model, val_series_rw, device: torch.device) -> torch.Tensor:
        constant = model.params[0]

        if not self.do_first_diff:
            if model.ar_lags is None:
                return constant
        
            lags = len(model.ar_lags)
            lag_params = torch.flip(torch.tensor(model.params[1:], dtype=torch.float32, device=device), [0])
            assert len(lag_params) == lags

            val_preds = torch.zeros(val_series_rw.shape[0], val_series_rw.shape[1], self.output_length, dtype=torch.float32, device=device)
            for t in range(self.output_length):
                if t == 0:
                    val_preds[:, :, t] = (val_series_rw[:, :, -lags - self.output_length:-self.output_length] * lag_params).sum(dim=-1) 
                elif t < lags:
                    val_preds[:, :, t] = (val_series_rw[:, :, -lags + t - self.output_length:-self.output_length] * lag_params[:-t]).sum(dim=-1) + (val_preds[:, :, :t] * lag_params[-t:]).sum(dim=-1)
                else:
                    val_preds[:, :, t] = (val_preds[:, :, t-lags:t] * lag_params).sum(dim=-1)
                val_preds[:, :, t] = val_preds[:, :, t] + constant 
            return val_preds
        else:
            if self.numdiff == 1:
                if model.ar_lags is None:
                    deltas = torch.arange(1, self.output_length + 1, 1, device=device) * constant
                    val_preds = val_series_rw[:, :, self.input_length - 1][:, :, None] + deltas
                    return val_preds
                
                lags = len(model.ar_lags)
                lag_params = torch.flip(torch.tensor(model.params[1:], dtype=torch.float32, device=device), [0])
                assert len(lag_params) == lags

                val_diffs = torch.diff(val_series_rw, 1, dim=-1).to(device)
                val_series_rw = val_series_rw.to(device)
                val_diff_preds = torch.zeros(val_series_rw.shape[0], val_series_rw.shape[1], self.output_length, dtype=torch.float32, device=device)
                for t in range(self.output_length):
                    if t == 0:
                        val_diff_preds[:, :, t] = (val_diffs[:, :, -lags - self.output_length:-self.output_length] * lag_params).sum(dim=-1)
                    elif t < lags:
                        val_diff_preds[:, :, t] = (val_diffs[:, :, -lags + t - self.output_length:-self.output_length] * lag_params[:-t]).sum(dim=-1) + (val_diff_preds[:, :, :t] * lag_params[-t:]).sum(dim=-1)
                    else:
                        val_diff_preds[:, :, t] = (val_diff_preds[:, :, t-lags:t] * lag_params).sum(dim=-1)
                    val_diff_preds[:, :, t] = val_diff_preds[:, :, t] + constant 
                val_preds = val_series_rw[:, :, self.input_length - 1, None] + torch.cumsum(val_diff_preds, dim=-1)
                return val_preds
            elif self.numdiff == 2:
                
                lags = len(model.ar_lags)
                lag_params = torch.flip(torch.tensor(model.params[1:], dtype=torch.float32, device=device), [0])
                assert len(lag_params) == lags

                val_diffs = torch.diff(val_series_rw, 2, dim=-1)
                val_diff_preds = torch.zeros(val_series_rw.shape[0], val_series_rw.shape[1], self.output_length, dtype=torch.float32, device=device)
                for t in range(self.output_length):
                    if t == 0:
                        val_diff_preds[:, :, t] = (val_diffs[:, :, -lags - self.output_length:-self.output_length] * lag_params).sum(dim=-1)
                    elif t < lags:
                        val_diff_preds[:, :, t] = (val_diffs[:, :, -lags + t - self.output_length:-self.output_length] * lag_params[:-t]).sum(dim=-1) + (val_diff_preds[:, :, :t] * lag_params[-t:]).sum(dim=-1)
                    else:
                        val_diff_preds[:, :, t] = (val_diff_preds[:, :, t-lags:t] * lag_params).sum(dim=-1)
                    val_diff_preds[:, :, t] = val_diff_preds[:, :, t] + constant 
                val_first_diff = val_series_rw[:, :, self.input_length - 1, None] - val_series_rw[:, :, self.input_length - 2, None]+ torch.cumsum(val_diff_preds, dim=-1)
                val_preds = val_series_rw[:, :, self.input_length - 1, None] + torch.cumsum(val_first_diff, dim=-1)
                # print(val_preds.shape)
                return val_preds


    def _test_loss_acc_stack(self, model, val_series_rw, device: torch.device, return_prediction=False, split=5):
        all_mse = 0
        all_mae = 0
        val_preds = []
        for i in range(split):
            val_series_rw_split = val_series_rw[:,i*val_series_rw.shape[0]//split:(i+1)*val_series_rw.shape[0]//split,:]
            val_series_rw_split = val_series_rw_split.to(device)
            val_pred = self._predict_stack(model, val_series_rw_split, device).to(torch.device("cpu"))
            all_mse += ((val_pred - val_series_rw_split[:, :, -self.output_length:])**2).mean().item()
            all_mae += torch.abs(val_pred - val_series_rw_split[:, :, -self.output_length:]).mean().item()
            val_preds.append(val_pred)
        all_mse /= split
        all_mae /= split
        val_preds = torch.cat(val_preds, dim=1)
        print(val_preds.shape)
        if return_prediction:
            return (
                all_mse, 
                all_mae,
                val_preds,
                val_series_rw[:, :, -self.output_length:]
            )
        return (
            all_mse, 
            all_mae
        )
       
    
    def test_loss_acc_df(self, scaled_test_df, device: torch.device, numdiff: int, return_prediction=False, model = None):
        if not model:
            model = self.model
        self.numdiff = numdiff
        test_series_rw = _unfold_df(scaled_test_df, device, self.input_length, self.output_length)
        # print(test_series_rw.shape)
        return self._test_loss_acc_stack(model, test_series_rw, device, return_prediction)

    def predict(self, input) -> torch.Tensor:
        preds = self.model.append(input.flatten(), refit=False).forecast(steps=self.output_length)
        return torch.tensor(preds)
    

