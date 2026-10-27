import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
import random
import os

def add_month(date, months=1):
    new_month = (date.month + months - 1) % 12 + 1
    new_year = date.year + (date.month + months - 1) // 12
    return date.replace(year=new_year, month=new_month)

def add_month_to_string_time(str_date, months=1):
    date = str(pd.to_datetime(str_date))
    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    new_date = add_month(date, months=months)
    return new_date.strftime("%Y-%m-%d")

def get_data_baseline(symbol, start_date, end_date, lookback=0, lookforward=0):
    data_baseline = np.load("../data/sp500/baseline_data_sp500.npy", allow_pickle=True).item()
    df = data_baseline[symbol]
    mask = (df.index >= start_date) & (df.index <= end_date)
    df_data = df.loc[mask]
    if lookback:
        concat_df = df.loc[:df_data.index[0]].tail(lookback)
        df_data = pd.concat([df_data, concat_df], axis=0).sort_index().drop_duplicates()
    if lookforward:
        concat_df = df.loc[df_data.index[-1]:].head(lookforward+1)
        df_data = pd.concat([df_data, concat_df], axis=0).sort_index().drop_duplicates()
    df_data = df_data.astype(float)
    return df_data

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
