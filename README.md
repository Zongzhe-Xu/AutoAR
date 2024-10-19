This is the official implementation of AutoAR that is introduced in this paper.
# Setup
- Raw data, download here: https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy (download the `all_six_dataset.zip` file)
- After downloading the data, put the folder all_six_dataset under /data/ts_datasets/.

-Set up conda environment by running:
```
conda create --name autoar --file requirements.txt
conda activate autoar
```

# Data
All datasets used up to this point are forecasting datasets.

Raw data consists of csv files where each file corresponds to a single dataset, the columns are channels, and the rows are timesteps. The contents should include
- ETT-small datasets (4 of them): ETTh1, ETTh2, ETTm1, ETTm2
- Electricity
- Illness
- Traffic
- Weather

# Experiments

To replicate results for ETTh1 with a forcasting length of 96, simply run
```
python3 autoar.py --dataset ETTh1_96 --new_metric --kpss
```
***--kpss***: Use KPSS test to determine the num_diff for the dataset provided. Set num_diff = 0 if not provided.

***--new_metric***: Use Bayesian Information Criteria (BIC) for choosing the input length. Use validation set for selection if not provided

***--dataset***: The dataset argument is formatted as "dataset name"_"forcasting length". For example, to run on ETTh1 with forecasting length of 720, use ETTh1_720.

***--use_ols***: Whether to use OLS for fitting the linear model. Default to True.
