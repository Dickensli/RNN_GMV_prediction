import os
import argparse

import tensorflow as tf
from model import Model
from input_pipe import InputPipe
from make_features import run
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

def pred(city_list = {1,  2,   3,   4,   5,   6,   7,   8,   9,  10,  12,  13,  14,
              15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  28, 29,
              32,  33,  34,  35,  36,  38,  39,  41,  44,  45,  46,  47,  48,
              50,  53,  58,  62,  63,  81,  82,  83,  84,  85,  86,  87,  88,
              89,  90,  92, 102, 105, 106, 118, 132, 133, 134, 135, 138, 142, 143,
              145, 153, 154, 157, 158, 159, 160, 173, 283} - {4, 11, 31}, **args):
    infer_y_origin = run(city_list=city_list)
    prev = infer_y_origin.groupby('city_id').tail(1).reset_index(drop=True)['total_no_call_order_cnt']
    paths = [p for p in tf.train.get_checkpoint_state(os.path.join('data/cpt', 's32')).all_model_checkpoint_paths]
    preds, loss = predict_loss(prev, paths)
    print(loss)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('--city_large', default=False, action='store_true', help='Whether to use large cities only')    
    parser.add_argument('--city_predict', default=False, action='store_true', help='Whether to predict 30 cities')     
    args = parser.parse_args()

    param_dict = dict(vars(args))
    param_dict['city_list'] = set(range(1, 357)) - {31, 181, 204, 205, 236, 237, 238, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 316}
    if args.city_large:
        param_dict['city_list'] = {1,  2,   3,   4,   5,   6,   7,   8,   9,  10,  12,  13,  14,
              15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  28, 29,
              32,  33,  34,  35,  36,  38,  39,  41,  44,  45,  46,  47,  48,
              50,  53,  58,  62,  63,  81,  82,  83,  84,  85,  86,  87,  88,
              89,  90,  92, 102, 105, 106, 118, 132, 133, 134, 135, 138, 142, 143,
              145, 153, 154, 157, 158, 159, 160, 173, 283} - {4, 11, 31}
    if args.city_predict:
        param_dict['city_list'] = {2,3,5,6,7,9,10,15,16,21,22,23,25,26,28,29,32,34,35,36,38,39,41,50,53,63,105,118,134,283}
    pred(**param_dict)