import h5py
import time
import pickle as pkl
import pandas as pd
import numpy as np
import os.path
import os
import argparse
import logging
import datetime

import extractor
from feeder import VarFeeder
import numba
from typing import Tuple, Dict, Collection, List
from feature_server import FeatureServer

log = logging.getLogger('makeFeatures')

def read_all(city_list, path_city_day, path_weather_forecast):
    '''
    return :
        [train_x, train_embed_weekday, train_embed_month,
     train_embed_city, train_real_city, train_y_origin]
     
    [val_x, val_embed_weekday, val_embed_month,
     val_embed_city, val_real_city, val_y_origin]
     
    [infer_x, infer_embed_weekday, infer_embed_month,
     infer_embed_city, infer_city_map, infer_y_origin]
     
    city_max, city_min, train_mean, train_std
    '''
    gen_feas = FeatureServer(city=city_list,
                         path_city_day=path_city_day,
                         path_weather_forecast=path_weather_forecast,
                         begin_train_day=datetime.datetime.strftime(datetime.date(2017, 4, 1), '%Y-%m-%d'),
                         end_train_day=datetime.datetime.strftime(datetime.date(2018, 7, 4), '%Y-%m-%d'),
                         begin_val_day=datetime.datetime.strftime(datetime.date(2018, 7, 5), '%Y-%m-%d'),
                         end_val_day=datetime.datetime.strftime(datetime.date(2018, 7, 26), '%Y-%m-%d'),
                         begin_infer_day=datetime.datetime.strftime(datetime.date(2018, 7, 27), '%Y-%m-%d'))
    return gen_feas.gen_whole_data(2)

@numba.jit(nopython=True)
def single_autocorr(series, lag):
    """
    Autocorrelation for single data series
    :param series: usage series
    :param lag: lag, days
    :return:
    """
    s1 = series[lag:]
    s2 = series[:-lag]
    if s1.shape[0] == 0 or s2.shape[0] == 0:
        return 0
    ms1 = np.mean(s1)
    ms2 = np.mean(s2)
    ds1 = s1 - ms1
    ds2 = s2 - ms2
    divider = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))
    return np.sum(ds1 * ds2) / divider if divider != 0 else 0


@numba.jit(nopython=True)
def batch_autocorr(data, lag):
    """
    Calculate autocorrelation for batch (many time series at once)
    :param data: Time series, shape [n_vm, n_time]
    :param lag: Autocorrelation lag
    :return: autocorrelation, shape [n_series]. If series is too short (support less than threshold),
    autocorrelation value is NaN
    """
    n_series = data.shape[0]
    n_days = data.shape[1]
    corr = np.empty(n_series, dtype=np.float64)
    for i in range(n_series):
        series = data[i]
        c_minus1 = single_autocorr(series, lag)
        c = single_autocorr(series, lag-1)
        c_plus1 = single_autocorr(series, lag+1)
        # Average value between exact lag and two nearest neighborhs for smoothness
        corr[i] = 0.5 * c_minus1 + 0.25 * c + 0.25 * c_plus1
    return corr

def normalize(values: np.ndarray):
    return (values - values.mean()) / np.std(values)

def run(city_path='/nfs/isolation_project/intern/project/lihaocheng/city_forcast/city_day_features_to_yesterday.gbk.csv', 
        weafor_path = '/nfs/isolation_project/intern/project/lihaocheng/city_forcast/weather_forecast.csv',
        datadir='data',
        city_large = True, **args):

    start_time = time.time()
    city_list = set(range(1, 357)) - {31, 181, 204, 205, 236, 237, 238, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 316}
    if city_large:
        city_list = {1,  2,   3,   4,   5,   6,   7,   8,   9,  10,  12,  13,  14,
              15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  29,
              32,  33,  34,  35,  36,  38,  39,  41,  44,  45,  46,  47,  48,
              50,  53,  58,  62,  63,  81,  82,  83,  84,  85,  86,  87,  88,
              89,  90,  92, 102, 105, 106, 132, 133, 134, 135, 138, 142, 143,
              145, 153, 154, 157, 158, 159, 160, 173} - {4, 11, 31}
    
    # Get the data
    [train_x, train_embed_weekday, train_embed_month,
     train_embed_city, train_real_city, train_y_origin],\
    [val_x, val_embed_weekday, val_embed_month,
     val_embed_city, val_real_city, val_y_origin],\
    [infer_x, infer_embed_weekday, infer_embed_month,
     infer_embed_city, infer_city_map, infer_y_origin],\
    city_max, city_min, train_mean, train_std = read_all(city_list, city_path, weafor_path)
    
    log.debug("complete generating df_cpu_max and df_cpu_num, time elapse = %S", time.time() - start_time)
   
    train_total = pd.concat([train_x, train_y_origin], axis=1)
    train_features = [pd.DataFrame() for city in city_list]
    train_y = [pd.DataFrame() for city in city_list]
    y_mean = list()
    y_std = list()
    month_autocorr = dict()
    week_autocorr = dict()
    
    # Make train features
    attrs = ['online_time', 'total_finish_order_cnt', 'total_gmv', 'strive_order_cnt', 'total_no_call_order_cnt']
    dfs = [pd.DataFrame() for i in range(5)]
    for i, city in enumerate(city_list):
        for idx, attr in enumerate(attrs):
            per_city = train_total[train_total['city_id'] == city]
            train_features[i] = per_city[train_x.columns].values
            train_y[i] = per_city[train_y_origin.columns].apply(lambda x : np.log(x + 1))
            series = train_y[i].loc[:, [attr]]
            series.columns = [city]
            series = series.reset_index(drop=True)
            dfs[idx] = pd.concat([dfs[idx], series], axis=1)
    for idx, attr in enumerate(attrs):
        df = dfs[idx].T
        # monthly autocorrelation
        month = batch_autocorr(df.values, 30)

        # weekly autocorrelation
        week = batch_autocorr(df.values, 7)

        # Normalise all the things
        month_autocorr[attr] = normalize(np.nan_to_num(month))
        week_autocorr[attr] = normalize(np.nan_to_num(week))
        
    # Find train_y mean & std
    for i, per_city in enumerate(train_y):
        y_mean.append(per_city.mean())
        y_std.append(per_city.std())
        
    # Make val features
    val_total = pd.concat([val_x, val_y_origin], axis=1)
    val_features = list()
    val_y = list()
    for city in city_list:
        per_city = val_total[val_total['city_id'] == city]
        val_features.append(per_city[val_x.columns].values)
        val_y.append(per_city[val_y_origin.columns].apply(lambda x : np.log(x + 1)))
    
    # Make infer features
    infer_x = infer_x.drop(['city_id'], axis=1)
    infer_total = pd.concat([infer_x, infer_y_origin], axis=1)
    infer_features = list()
    infer_y = list()
    for city in city_list:
        per_city = infer_total[infer_total['city_id'] == city]
        infer_features.append(per_city[infer_x.columns].values)  
        infer_y.append(per_city[infer_y_origin.columns].apply(lambda x : np.log(x + 1)))

    # Make time-dependent features
    time_period = 7 / (2 * np.pi)
    train_dow_norm = train_embed_weekday / time_period
    val_dow_norm = val_embed_weekday / time_period
    infer_dow_norm = infer_embed_weekday / time_period
    time_period = 12 / (2 * np.pi)
    train_dom_norm = train_embed_month / time_period
    val_dom_norm = val_embed_month / time_period
    infer_dom_norm = infer_embed_month / time_period    
    
    train_dow = np.stack([np.cos(train_dow_norm), np.sin(train_dow_norm), 
                          np.cos(train_dom_norm), np.sin(train_dom_norm)], axis=-1)
    val_dow = np.stack([np.cos(val_dow_norm), np.sin(val_dow_norm),
                       np.cos(val_dom_norm), np.sin(val_dom_norm)], axis=-1)
    infer_dow = np.stack([np.cos(infer_dow_norm), np.sin(infer_dow_norm),
                        np.cos(infer_dom_norm), np.sin(infer_dom_norm)], axis=-1)
    
    # Assemble final output
    tensors = dict(
        train_x=train_features,
        val_x=val_features,
        infer_x=infer_features,
        
        train_dow=train_dow,
        val_dow=val_dow,
        infer_dow=infer_dow,
        
        train_y=[df['total_no_call_order_cnt'] for df in train_y],
        val_y=[df['total_no_call_order_cnt'] for df in val_y],
        infer_y=[df['total_no_call_order_cnt'] for df in infer_y],
        
        train_time=train_features[0].shape[0],
        val_time=val_features[0].shape[0],
        infer_time=infer_features[0].shape[0],
        
        month_autocorr=month_autocorr['total_no_call_order_cnt'],
        week_autocorr=week_autocorr['total_no_call_order_cnt'],
        
        cities=np.arange(len(city_list)),
        mean=[per['total_no_call_order_cnt'] for per in y_mean],
        std=[per['total_no_call_order_cnt'] for per in y_std]
    )
    plain = dict(
    )

    # Store data to the disk
    VarFeeder(os.path.join(datadir, 'vars'), tensors, plain)
    