from abc import ABC
from torch.utils.data import Dataset
from dataclasses import dataclass
import numpy.typing as npt
from typing import Optional, List, Tuple, Union

import os
from typing import Optional

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import DataLoader


@dataclass
class TimeseriesData:
    timeseries: npt.NDArray = None
    forecast: npt.NDArray = None
    labels: Union[npt.NDArray, int, str] = None
    input_mask: npt.NDArray = None
    metadata: dict = None
    name: str = None

@dataclass
class DataSplits:
    train: npt.NDArray = None
    val: npt.NDArray = None
    test: npt.NDArray = None

class TaskDataset(ABC, Dataset):
    def __init__(self):
        super(TaskDataset, self).__init__()
    
    def _read_data(self) -> TimeseriesData:
        return NotImplementedError
    
    def __len__(self):
        return NotImplementedError
    
    def __getitem__(self, idx):
        return NotImplementedError

    def plot(self, idx):
        return NotImplementedError
    
    def _check_and_remove_nans(self):
        return NotImplementedError

    def _subsample(self):
        return NotImplementedError


class LongForecastingDataset(TaskDataset):
    def __init__(self, 
                 seq_len : int = 512,
                 forecast_horizon : int = 96,
                 full_file_path_and_name: str = '../TimeseriesDatasets/forecasting/autoformer/ETTh1.csv',
                 data_split : str = 'train',
                 target_col : Optional[str] = 'OT', 
                 scale : bool = True,
                 data_stride_len : int = 1,
                 task_name : str = 'long-horizon-forecasting',
                 train_ratio : float = 0.6,
                 val_ratio : float = 0.1,
                 test_ratio : float = 0.3,
                 output_type : str = 'univariate',
                 random_seed : int = 42,
                 multi_channel : bool = False,
                 **kwargs,
                 ):
        super(LongForecastingDataset, self).__init__()
        """
        Parameters
        ----------
        seq_len : int
            Length of the input sequence.
        forecast_horizon : int
            Length of the prediction sequence.
        full_file_path_and_name : str
            Name of the dataset.
        data_split : str
            Split of the dataset, 'train', 'val' or 'test'.
        target_col : str
            Name of the target column. 
            If None, the target column is the last column.
        scale : bool
            Whether to scale the dataset.
        data_stride_len : int
            Stride length when generating consecutive 
            time-series windows. 
        task_name : str
            The task that the dataset is used for. One of
            'forecasting', 'pre-training', or , 'imputation'.
        train_ratio : float
            Ratio of the training set.
        val_ratio : float
            Ratio of the validation set.
        test_ratio : float
            Ratio of the test set.
        output_type : str
            The type of the output. One of 'univariate' 
            or 'multivariate'. If multivariate, either the 
            target column must be specified or the dataset
            is flattened along the channel dimension.
        random_seed : int
            Random seed for reproducibility.
        """
        # print('full_file_path_and_name', full_file_path_and_name)
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.full_file_path_and_name = full_file_path_and_name

        self.dataset_name = full_file_path_and_name.split('/')[-1][:-4]

        self.data_split = data_split
        self.target_col = target_col
        self.scale = scale
        self.data_stride_len = data_stride_len
        self.task_name = task_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.output_type = output_type
        self.random_seed = random_seed
        self.multi_channel = multi_channel

        # Input checking
        self._check_inputs()
        
        # Read data
        self._read_data()

    def _check_inputs(self):
        assert self.data_split in ['train', 'test', 'val'],\
            "data_split must be one of 'train', 'test' or 'val'"
        assert self.task_name in ['long-horizon-forecasting', 'pre-training', 'imputation'],\
            "task_name must be one of 'long-horizon-forecasting', 'pre-training', 'imputation'"
        assert self.output_type in ['univariate', 'multivariate'],\
            "output_type must be one of 'univariate' or 'multivariate'"
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1,\
            "train_ratio + val_ratio + test_ratio must be equal to 1"
        assert self.data_stride_len > 0,\
            "data_stride_len must be greater than 0"
            
    def __repr__(self):
        repr = f"LongForecastingDataset(dataset_name={self.dataset_name}," + \
            f"length_timeseries={self.length_timeseries}," + \
            f"length_dataset={self.__len__()}," + \
            f"n_channels={self.n_channels}," + \
            f"seq_len={self.seq_len}," + \
            f"forecast_horizon={self.forecast_horizon}," + \
            f"data_split={self.data_split}," + \
            f"target_col={self.target_col}," + \
            f"scale={self.scale}," + \
            f"data_stride_len={self.data_stride_len}," + \
            f"task_name={self.task_name}," + \
            f"train_ratio={self.train_ratio}," + \
            f"val_ratio={self.val_ratio}," + \
            f"test_ratio={self.test_ratio}," + \
            f"output_type={self.output_type})"
        return repr

    def _get_borders(self):
        ### This is for the AutoFormer datasets
        remaining_autoformer_datasets = [
            'electricity', 'exchange_rate', 
            'national_illness', 'traffic', 
            'weather']
        
        if 'ETTm' in self.dataset_name:
            n_train = 12 * 30 * 24 * 4
            n_val = 4 * 30 * 24 * 4
            n_test = 4 * 30 * 24 * 4
               
        elif 'ETTh' in self.dataset_name:
            n_train = 12 * 30 * 24
            n_val = 4 * 30 * 24
            n_test = 4 * 30 * 24

        # elif 'illness' in self.dataset_name:
        #     n_train = 617
        #     n_val = 74
        #     n_test = 170

        elif self.dataset_name in remaining_autoformer_datasets:
            n_train = int(self.train_ratio*self.length_timeseries_original)
            n_test = int(self.test_ratio*self.length_timeseries_original)
            n_val = self.length_timeseries_original - n_train - n_test

        train_end = n_train
        val_start = train_end - self.seq_len
        val_end = n_train + n_val
        self.test_start = val_end - self.seq_len
        self.test_end = self.test_start + n_test + self.seq_len
            
        return DataSplits(train=slice(0, train_end), 
                          val=slice(val_start, val_end), 
                          test=slice(self.test_start, self.test_end))
        
    def _read_data(self) -> TimeseriesData:
        self.scaler = StandardScaler()
        df = pd.read_csv(self.full_file_path_and_name)
        self.length_timeseries_original = df.shape[0]
        self.n_channels = df.shape[1] - 1

        # Following would only work for AutoFormer datasets
        df.drop(columns=['date'], inplace=True)
        # df = df.infer_objects(copy=False).interpolate(method='cubic')
        # MANUAL
        df = df.infer_objects().interpolate(method='cubic')

        if self.target_col in list(df.columns) and self.output_type == 'univariate':
            df = df[[self.target_col]]
            self.n_channels = 1
        elif self.target_col is None and self.output_type == 'univariate' and self.n_channels > 1:
            raise ValueError("target_col must be specified if output_type\
                              is 'univariate' for multi-channel datasets")
        
        data_splits = self._get_borders()
        
        if self.scale:
            train_data = df[data_splits.train]
            self.scaler.fit(train_data.values)
            df = self.scaler.transform(df.values)
        else:
            df = df.values
        
        if self.data_split == 'train':
            self.data = df[data_splits.train, :]
        elif self.data_split == 'val':
            self.data = df[data_splits.val, :]
        elif self.data_split == 'test':
            self.data = df[data_splits.test, :]        

        self.length_timeseries = self.data.shape[0]

    def __getitem__(self, raw_index):
        t_length = (self.length_timeseries - self.seq_len - self.forecast_horizon) // self.data_stride_len + 1
        channel_index, index = raw_index // t_length, raw_index % t_length
        # channel_index, index = 5, raw_index % t_length
        seq_start = self.data_stride_len*index
        seq_end = seq_start + self.seq_len
        input_mask = np.ones(self.seq_len)
        
        if self.task_name == 'long-horizon-forecasting':
            pred_end = seq_end + self.forecast_horizon

            if pred_end > self.length_timeseries:
                pred_end = self.length_timeseries
                seq_end = seq_end - self.forecast_horizon
                seq_start = seq_end - self.seq_len

            # return TimeseriesData(
            #         timeseries=self.data[seq_start:seq_end, :].T,
            #         forecast=self.data[seq_end:pred_end, :].T,
            #         input_mask=input_mask,
            #         name=self.dataset_name,
            #         metadata={
            #             'target_col': self.target_col,
            #             'output_type': self.output_type,
            #         }
            #     )

            data1 = self.data[seq_start:seq_end, :].T
            data2 = self.data[seq_end:pred_end, :].T
            if self.multi_channel:
                return torch.tensor(data1, dtype=torch.float32), torch.tensor(data2, dtype=torch.float32)
            else:
                return torch.tensor(data1, dtype=torch.float32)[channel_index][None, :], torch.tensor(data2, dtype=torch.float32)[channel_index][None, :]
        
        elif self.task_name == 'pre-training' or self.task_name == 'imputation':
            if seq_end > self.length_timeseries:
                seq_end = self.length_timeseries
                seq_end = seq_end - self.seq_len
                        
            return TimeseriesData(
                timeseries=self.data[seq_start:seq_end, :].T,
                input_mask=input_mask,
                name=self.dataset_name,
                metadata={
                        'target_col': self.target_col,
                        'output_type': self.output_type,
                        }
            )

    def __len__(self):
        if self.task_name == 'pre-training' or self.task_name == 'imputation':
            return (self.length_timeseries - self.seq_len) // self.data_stride_len + 1
        elif self.task_name == 'long-horizon-forecasting':
            if self.multi_channel:
                return ((self.length_timeseries - self.seq_len - self.forecast_horizon) // self.data_stride_len + 1)
            else:
                return ((self.length_timeseries - self.seq_len - self.forecast_horizon) // self.data_stride_len + 1) * self.n_channels
            
    def plot(self, idx, channel=0):
        timeseries_data = self.__getitem__(idx)
        # forecast = timeseries_data.forecast[channel, :]
        # timeseries = timeseries_data.timeseries[channel, :]
        forecast = timeseries_data[1][channel, :]
        timeseries = timeseries_data[0][channel, :]
        # input_mask = timeseries_data.input_mask

        plt.title(f"idx={idx}", fontsize=18)
        plt.plot(
            np.arange(self.seq_len),
            timeseries.flatten(),
            label='Time-series', c='darkblue'
        )
        if self.task_name == 'long-horizon-forecasting':
            plt.plot(
                np.arange(self.seq_len, self.seq_len + self.forecast_horizon), 
                forecast.flatten(),
                label='Forecast', c='red', linestyle='--'
            )
        
        plt.xlabel('Time', fontsize=18)
        plt.ylabel('Value', fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=18)
        plt.show()
        
    
    def check_and_remove_nans(self):
        return NotImplementedError
    
def _get_forecasts(examples):
    forecasts = [torch.from_numpy(example.forecast) for example in examples]
    forecasts = torch.stack(forecasts)
    return forecasts
    
def _collate_fn_basic(examples):
    examples = list(filter(lambda x: x is not None, examples))
    timeseries = [torch.from_numpy(example.timeseries) for example in examples]
    input_masks = [torch.from_numpy(example.input_mask) for example in examples]
    names = [example.name for example in examples]
    timeseries = torch.stack(timeseries)
    input_masks = torch.stack(input_masks)
    names = np.asarray(names)
    
    return TimeseriesData(timeseries=timeseries, 
                          input_mask=input_masks,
                          name=names)
    
def _collate_fn_forecasting(examples):
    # batch = _collate_fn_basic(examples)
    # batch.forecast = _get_forecasts(examples)
    # Alex Alteration
    # return batch.timeseries.view(-1, batch.timeseries.shape[-1]), [batch.forecast.view(-1, batch.forecast.shape[-1]), batch.input_mask.repeat(batch.timeseries.shape[1], 1)]
    x = torch.stack([e[0] for e in examples], dim=0)
    y = torch.stack([e[1] for e in examples], dim=0)
    return x, y
    
def get_timeseries_dataloaders(data_path: str, batch_size: int, seq_len=512, forecast_horizon=96, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, multi_channel=False):
    dataloaders = dict()
    for split in ['train', 'val', 'test']:
        dataset = LongForecastingDataset(
            full_file_path_and_name=data_path,
            output_type='multivariate',
            data_split=split,
            seq_len=seq_len,
            forecast_horizon=forecast_horizon,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            multi_channel=multi_channel
        )
        dataloaders[split] = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=split=='train',
            # num_workers=args.num_workers, 
            # pin_memory=args.pin_memory,
            collate_fn=_collate_fn_forecasting
        )
    return dataloaders['train'], dataloaders['val'], dataloaders['test']

def get_num_channels(dataset_name: str) -> int:
    if "ett" in dataset_name.lower():
        return 7 
    if "electricity" in dataset_name.lower():
        return 321 
    if "traffic" in dataset_name.lower():
        return 862 
    if "weather" in dataset_name.lower():
        return 21 
    if "exchange" in dataset_name.lower():
        return 8
    if "illness" in dataset_name.lower():
        return 7
    raise ValueError(f"Unrecognized dataset name: {dataset_name}")



class LongForecastingDataset_date(TaskDataset):
    def __init__(self, 
                 seq_len : int = 512,
                 forecast_horizon : int = 96,
                 full_file_path_and_name: str = '../TimeseriesDatasets/forecasting/autoformer/ETTh1.csv',
                 data_split : str = 'train',
                 target_col : Optional[str] = 'OT', 
                 scale : bool = True,
                 data_stride_len : int = 1,
                 task_name : str = 'long-horizon-forecasting',
                 train_ratio : float = 0.6,
                 val_ratio : float = 0.1,
                 test_ratio : float = 0.3,
                 output_type : str = 'univariate',
                 random_seed : int = 42,
                 multi_channel : bool = False,
                 **kwargs,
                 ):
        super(LongForecastingDataset, self).__init__()
        """
        Parameters
        ----------
        seq_len : int
            Length of the input sequence.
        forecast_horizon : int
            Length of the prediction sequence.
        full_file_path_and_name : str
            Name of the dataset.
        data_split : str
            Split of the dataset, 'train', 'val' or 'test'.
        target_col : str
            Name of the target column. 
            If None, the target column is the last column.
        scale : bool
            Whether to scale the dataset.
        data_stride_len : int
            Stride length when generating consecutive 
            time-series windows. 
        task_name : str
            The task that the dataset is used for. One of
            'forecasting', 'pre-training', or , 'imputation'.
        train_ratio : float
            Ratio of the training set.
        val_ratio : float
            Ratio of the validation set.
        test_ratio : float
            Ratio of the test set.
        output_type : str
            The type of the output. One of 'univariate' 
            or 'multivariate'. If multivariate, either the 
            target column must be specified or the dataset
            is flattened along the channel dimension.
        random_seed : int
            Random seed for reproducibility.
        """
        # print('full_file_path_and_name', full_file_path_and_name)
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.full_file_path_and_name = full_file_path_and_name

        self.dataset_name = full_file_path_and_name.split('/')[-1][:-4]

        self.data_split = data_split
        self.target_col = target_col
        self.scale = scale
        self.data_stride_len = data_stride_len
        self.task_name = task_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.output_type = output_type
        self.random_seed = random_seed
        self.multi_channel = multi_channel

        # Input checking
        self._check_inputs()
        
        # Read data
        self._read_data()

    def _check_inputs(self):
        assert self.data_split in ['train', 'test', 'val'],\
            "data_split must be one of 'train', 'test' or 'val'"
        assert self.task_name in ['long-horizon-forecasting', 'pre-training', 'imputation'],\
            "task_name must be one of 'long-horizon-forecasting', 'pre-training', 'imputation'"
        assert self.output_type in ['univariate', 'multivariate'],\
            "output_type must be one of 'univariate' or 'multivariate'"
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1,\
            "train_ratio + val_ratio + test_ratio must be equal to 1"
        assert self.data_stride_len > 0,\
            "data_stride_len must be greater than 0"
            
    def __repr__(self):
        repr = f"LongForecastingDataset(dataset_name={self.dataset_name}," + \
            f"length_timeseries={self.length_timeseries}," + \
            f"length_dataset={self.__len__()}," + \
            f"n_channels={self.n_channels}," + \
            f"seq_len={self.seq_len}," + \
            f"forecast_horizon={self.forecast_horizon}," + \
            f"data_split={self.data_split}," + \
            f"target_col={self.target_col}," + \
            f"scale={self.scale}," + \
            f"data_stride_len={self.data_stride_len}," + \
            f"task_name={self.task_name}," + \
            f"train_ratio={self.train_ratio}," + \
            f"val_ratio={self.val_ratio}," + \
            f"test_ratio={self.test_ratio}," + \
            f"output_type={self.output_type})"
        return repr

    def _get_borders(self):
        ### This is for the AutoFormer datasets
        remaining_autoformer_datasets = [
            'electricity', 'exchange_rate', 
            'national_illness', 'traffic', 
            'weather']
        
        if 'ETTm' in self.dataset_name:
            n_train = 12 * 30 * 24 * 4
            n_val = 4 * 30 * 24 * 4
            n_test = 4 * 30 * 24 * 4
               
        elif 'ETTh' in self.dataset_name:
            n_train = 12 * 30 * 24
            n_val = 4 * 30 * 24
            n_test = 4 * 30 * 24

        # elif 'illness' in self.dataset_name:
        #     n_train = 617
        #     n_val = 74
        #     n_test = 170

        elif self.dataset_name in remaining_autoformer_datasets:
            n_train = int(self.train_ratio*self.length_timeseries_original)
            n_test = int(self.test_ratio*self.length_timeseries_original)
            n_val = self.length_timeseries_original - n_train - n_test

        train_end = n_train
        val_start = train_end - self.seq_len
        val_end = n_train + n_val
        self.test_start = val_end - self.seq_len
        self.test_end = self.test_start + n_test + self.seq_len
            
        return DataSplits(train=slice(0, train_end), 
                          val=slice(val_start, val_end), 
                          test=slice(self.test_start, self.test_end))
        
    def _read_data(self) -> TimeseriesData:
        self.scaler = StandardScaler()
        df = pd.read_csv(self.full_file_path_and_name)
        self.length_timeseries_original = df.shape[0]
        self.n_channels = df.shape[1] - 1

        # Following would only work for AutoFormer datasets
        df.drop(columns=['date'], inplace=True)
        # df = df.infer_objects(copy=False).interpolate(method='cubic')
        # MANUAL
        df = df.infer_objects().interpolate(method='cubic')

        if self.target_col in list(df.columns) and self.output_type == 'univariate':
            df = df[[self.target_col]]
            self.n_channels = 1
        elif self.target_col is None and self.output_type == 'univariate' and self.n_channels > 1:
            raise ValueError("target_col must be specified if output_type\
                              is 'univariate' for multi-channel datasets")
        
        data_splits = self._get_borders()
        
        if self.scale:
            train_data = df[data_splits.train]
            self.scaler.fit(train_data.values)
            df = self.scaler.transform(df.values)
        else:
            df = df.values
        
        if self.data_split == 'train':
            self.data = df[data_splits.train, :]
        elif self.data_split == 'val':
            self.data = df[data_splits.val, :]
        elif self.data_split == 'test':
            self.data = df[data_splits.test, :]        

        self.length_timeseries = self.data.shape[0]

    def __getitem__(self, raw_index):
        t_length = (self.length_timeseries - self.seq_len - self.forecast_horizon) // self.data_stride_len + 1
        channel_index, index = raw_index // t_length, raw_index % t_length
        # channel_index, index = 5, raw_index % t_length
        seq_start = self.data_stride_len*index
        seq_end = seq_start + self.seq_len
        input_mask = np.ones(self.seq_len)
        
        if self.task_name == 'long-horizon-forecasting':
            pred_end = seq_end + self.forecast_horizon

            if pred_end > self.length_timeseries:
                pred_end = self.length_timeseries
                seq_end = seq_end - self.forecast_horizon
                seq_start = seq_end - self.seq_len

            # return TimeseriesData(
            #         timeseries=self.data[seq_start:seq_end, :].T,
            #         forecast=self.data[seq_end:pred_end, :].T,
            #         input_mask=input_mask,
            #         name=self.dataset_name,
            #         metadata={
            #             'target_col': self.target_col,
            #             'output_type': self.output_type,
            #         }
            #     )

            data1 = self.data[seq_start:seq_end, :].T
            data2 = self.data[seq_end:pred_end, :].T
            if self.multi_channel:
                return torch.tensor(data1, dtype=torch.float32), torch.tensor(data2, dtype=torch.float32)
            else:
                return torch.tensor(data1, dtype=torch.float32)[channel_index][None, :], torch.tensor(data2, dtype=torch.float32)[channel_index][None, :]
        
        elif self.task_name == 'pre-training' or self.task_name == 'imputation':
            if seq_end > self.length_timeseries:
                seq_end = self.length_timeseries
                seq_end = seq_end - self.seq_len
                        
            return TimeseriesData(
                timeseries=self.data[seq_start:seq_end, :].T,
                input_mask=input_mask,
                name=self.dataset_name,
                metadata={
                        'target_col': self.target_col,
                        'output_type': self.output_type,
                        }
            )

    def __len__(self):
        if self.task_name == 'pre-training' or self.task_name == 'imputation':
            return (self.length_timeseries - self.seq_len) // self.data_stride_len + 1
        elif self.task_name == 'long-horizon-forecasting':
            if self.multi_channel:
                return ((self.length_timeseries - self.seq_len - self.forecast_horizon) // self.data_stride_len + 1)
            else:
                return ((self.length_timeseries - self.seq_len - self.forecast_horizon) // self.data_stride_len + 1) * self.n_channels
            
    def plot(self, idx, channel=0):
        timeseries_data = self.__getitem__(idx)
        # forecast = timeseries_data.forecast[channel, :]
        # timeseries = timeseries_data.timeseries[channel, :]
        forecast = timeseries_data[1][channel, :]
        timeseries = timeseries_data[0][channel, :]
        # input_mask = timeseries_data.input_mask

        plt.title(f"idx={idx}", fontsize=18)
        plt.plot(
            np.arange(self.seq_len),
            timeseries.flatten(),
            label='Time-series', c='darkblue'
        )
        if self.task_name == 'long-horizon-forecasting':
            plt.plot(
                np.arange(self.seq_len, self.seq_len + self.forecast_horizon), 
                forecast.flatten(),
                label='Forecast', c='red', linestyle='--'
            )
        
        plt.xlabel('Time', fontsize=18)
        plt.ylabel('Value', fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=18)
        plt.show()
        
    
    def check_and_remove_nans(self):
        return NotImplementedError