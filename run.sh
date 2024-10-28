#!/bin/bash

DS = $1
# Time Series Forecasting: [dataset]_[horizon_length]; 
#       -- datasets: ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ECL', 'ILI', 'Traffic', 'Weather']
#       -- horizon_length:  [24, 36, 48, 60] for ILI
#                           [96, 192, 336, 720] for others 

python3 autoar.py --dataset $DS --new_metric --kpss