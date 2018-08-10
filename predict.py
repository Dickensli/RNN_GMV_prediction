import os
import tensorflow as tf
from model import Model
from input_pipe import InputPipe
from feeder import VarFeeder
from tqdm import trange
import collections
import pandas as pd
import numpy as np
from trainer import predict
from hparams import build_hparams
import hparams

from feature_server import FeatureServer
import datetime

def smape(true, pred):
    summ = np.abs(true) + np.abs(pred)
    smape = np.where(summ == 0, 0, np.abs(true - pred) / summ)
    return smape

def mae(true, pred):
    return np.abs(np.abs(true) - np.abs(pred))

def mean_smape(true, pred):
    raw_smape = smape(true, pred)
    masked_smape = np.ma.array(raw_smape, mask=np.isnan(raw_smape))
    return masked_smape.mean()

def mean_mae(true, pred):
    raw_mae = mae(true, pred)
    masked_mae = np.ma.array(raw_mae, mask=np.isnan(raw_mae))
    return masked_mae.mean()

def predict_loss(prev, paths):
    # prev: true value
    # paths: paths to the model weights
    t_preds = []
    for tm in range(3):
        tf.reset_default_graph()
        t_preds.append(predict(paths[-1:], build_hparams(hparams.params_s32),
                        n_models=3, target_model=tm, seed=5, batch_size=50, asgd=True))
    preds=sum(t_preds) /3
    # mean mae
    res = 0
    for idx in preds.index:
        res += np.abs(preds[idx] - prev[idx]) / prev[idx]
    res /= 72
    return preds, res

def run():
    city_list = [1,  2,   3,   4,   5,   6,   7,   8,   9,  10,  12,  13,  14,
                  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  29,
                  32,  33,  34,  35,  36,  38,  39,  41,  44,  45,  46,  47,  48,
                  50,  53,  58,  62,  63,  81,  82,  83,  84,  85,  86,  87,  88,
                  89,  90,  92, 102, 105, 106, 132, 133, 134, 135, 138, 142, 143,
                  145, 153, 154, 157, 158, 159, 160, 173]
    city_list = set(city_list) - {4, 11, 31}
    path_city_day = '/nfs/isolation_project/intern/project/lihaocheng/city_forcast/city_day_features_to_yesterday.gbk.csv'
    path_weather_forecast = '/nfs/isolation_project/intern/project/lihaocheng/city_forcast/weather_forecast.csv'
    gen_feas = FeatureServer(city=city_list,
                             path_city_day=path_city_day,
                             path_weather_forecast=path_weather_forecast,
                             begin_train_day=datetime.datetime.strftime(datetime.date(2017, 4, 1), '%Y-%m-%d'),
                             end_train_day=datetime.datetime.strftime(datetime.date(2018, 7, 4), '%Y-%m-%d'),
                             begin_val_day=datetime.datetime.strftime(datetime.date(2018, 7, 5), '%Y-%m-%d'),
                             end_val_day=datetime.datetime.strftime(datetime.date(2018, 7, 26), '%Y-%m-%d'),
                             begin_infer_day=datetime.datetime.strftime(datetime.date(2018, 7, 27), '%Y-%m-%d'))
    [train_x, train_embed_weekday, train_embed_month,
     train_embed_city, train_real_city, train_y_origin],\
    [val_x, val_embed_weekday, val_embed_month,
     val_embed_city, val_real_city, val_y_origin],\
    [infer_x, infer_embed_weekday, infer_embed_month,
     infer_embed_city, infer_city_map, infer_y_origin],\
    city_max, city_min, train_mean, train_std = gen_feas.gen_whole_data(2)

    prev = infer_y_origin.groupby('city_id').tail(1).reset_index(drop=True)['total_no_call_order_cnt']
    paths = [p for p in tf.train.get_checkpoint_state(os.path.join('data/cpt', 's32')).all_model_checkpoint_paths]
    paths = [p for p in tf.train.get_checkpoint_state(os.path.join('data/cpt', 's32')).all_model_checkpoint_paths]
    preds, loss = predict_loss(prev, paths)
    print(loss)
    
if __name__ == '__main__':
    run()