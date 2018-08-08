import tensorflow as tf

from feeder import VarFeeder
from enum import Enum
from typing import List, Iterable
import numpy as np
import pandas as pd
import os

class ModelMode(Enum):
    TRAIN = 0
    EVAL = 1,
    PREDICT = 2

class Split:
    def __init__(self, test_set: List[tf.Tensor], train_set: List[tf.Tensor], test_size: int, train_size: int):
        self.test_set = test_set
        self.train_set = train_set
        self.test_size = test_size
        self.train_size = train_size

class FakeSplitter:
    def __init__(self, tensors: List[tf.Tensor], n_splits, seed, test_sampling=1.0):
        total_vm = tensors[0].shape[0].value
        n_vm = int(round(total_vm * test_sampling))

        def mk_name(prefix, tensor):
            return prefix + '_' + tensor.name[:-2]

        def prepare_split(i):
            idx = tf.random_shuffle(tf.range(0, n_vm, dtype=tf.int32), seed + i)
            train_tensors = [tf.gather(tensor, idx, name=mk_name('shfl', tensor)) for tensor in tensors]
            if test_sampling < 1.0:
                sampled_idx = idx[:n_vm]
                test_tensors = [tf.gather(tensor, sampled_idx, name=mk_name('shfl_test', tensor)) for tensor in tensors]
            else:
                test_tensors = train_tensors
            return Split(test_tensors, train_tensors, n_vm, total_vm)

        self.splits = [prepare_split(i) for i in range(n_splits)]


class InputPipe:
    def cut_train(self, train_x, train_y, val_x, val_y, *args):
        """
        Cuts a segment of time series for training. Randomly chooses starting point.
        :param usage: usage timeseries
        :param args: pass-through data, will be appended to result
        :return: result of cut() + args
        """
        n_time = self.train_window
        # How much free space we have to choose starting day
        assert self.mode in [ModelMode.TRAIN, ModelMode.EVAL], "Invlid mode. mode should be chosen from [TRAIN, EVAL]"
        if self.mode == ModelMode.TRAIN:
            free_space = self.inp.train_time - n_time
        else:
            free_space = self.inp.val_time - n_time
        if self.verbose:
            lower_train_start = 0
            upper_train_start = free_space - 1
            print(f"Free space for training: {free_space} days.")
            print(f" Lower train {lower_train_start}")
            print(f" Upper train {upper_train_start}")

        # Random starting point
        offset = tf.random_uniform((), 0, free_space, dtype=tf.int64, seed=self.rand_seed)
        end = offset + n_time
        
        #offset = tf.Print(offset, [offset], 'offset')
        #end = tf.Print(end, [end], 'offset')
        
        if self.mode == ModelMode.TRAIN:
            x = train_x[offset: end]
            y = train_y[offset: end]
            dow = self.inp.train_dow[offset: end]
        else:
            x = val_x[offset: end]
            y = val_y[offset: end]
            dow = self.inp.val_dow[offset: end]
        
        # Cut all the things
        return (x, y, dow) + args
    
    def cut_infer(self, infer_x, infer_y, *args):
        """
        Cuts a segment of time series for training. Randomly chooses starting point.
        :param usage: usage timeseries
        :param args: pass-through data, will be appended to result
        :return: result of cut() + args
        """
        end = self.train_window
        offset = 0
        # Cut dow
        dow = self.inp.infer_dow[offset: end]
        x = infer_x[offset: end]
        y = infer_y[offset: end]

        # Cut all the things
        return (x, y, dow) + args

    def make_features(self, x, y, dow, cities, mean, std, month_autocorr, week_autocorr):
        """
        Main method. Assembles input data into final tensors
        """        
        # Combine all vm features into single tensor
        stacked_features = tf.stack([month_autocorr, week_autocorr])
        flat_vm_features = stacked_features
        vm_features = tf.expand_dims(flat_vm_features, 0)
        
        # Train features
        x_features = tf.concat([
            # [n_days, ] -> [n_days, 1]
            x,
            dow,
            # Stretch vm_features to all training days
            # [1, features] -> [n_days, features]
            tf.tile(vm_features, [self.train_window, 1])
        ], axis=1)

        # Test features
        y_features = y

        return x, x_features, y, y_features, cities, mean, std

    def __init__(self, datadir, inp: VarFeeder, features: Iterable[tf.Tensor], mode: ModelMode, n_epoch=None,
                 batch_size=20, runs_in_burst=1, verbose=True, train_window=8,
                 train_completeness_threshold=1, train_skip_first=0, rand_seed=None):
        """
        Create data preprocessing pipeline
        :param inp: Raw input data
        :param features: Features tensors (subset of data in inp)
        :param mode: Train/Predict/Eval mode selector
        :param n_epoch: Number of epochs. Generates endless data stream if None
        :param batch_size:
        :param runs_in_burst: How many batches can be consumed at short time interval (burst). Multiplicator for prefetch()
        :param verbose: Print additional information during graph construction
        :param predict_window: Number of timestamps to predict
        :param train_window: Use train_window timestamps for traning
        :param train_completeness_threshold: Percent of zero datapoints allowed in train timeseries.
        :param predict_completeness_threshold: Percent of zero datapoints allowed in test/predict timeseries.
        :param back_offset: Don't use back_offset days at the end of timeseries
        :param train_skip_first: Don't use train_skip_first days at the beginning of timeseries
        :param rand_seed:

        """
        self.inp = inp
        self.batch_size = batch_size
        self.rand_seed = rand_seed
        self.n_cities = inp.cities.get_shape()[0]

        if verbose:
            if mode == ModelMode.TRAIN:
                print("Mode:%s, data days:%d" % (mode, inp.train_time))
            if mode == ModelMode.EVAL:
                print("Mode:%s, data days:%d" % (mode, inp.val_time))
            if mode == ModelMode.PREDICT:
                print("Mode:%s, data days:%d" % (mode, inp.infer_time))

        self.train_window = train_window
        self.mode = mode
        self.verbose = verbose

        # Reserve more processing threads for eval/predict because of larger batches
        num_threads = 3 if mode == ModelMode.TRAIN else 6
        
        # Create dataset, transform features and assemble batches
        cutter = {ModelMode.TRAIN:self.cut_train, ModelMode.EVAL:self.cut_train, ModelMode.PREDICT:self.cut_infer}
        root_ds = tf.data.Dataset.from_tensor_slices(tuple(features)).repeat(n_epoch)
        batch = (root_ds
                 .map(cutter[mode])
                 .map(self.make_features, num_parallel_calls=num_threads)
                 .batch(batch_size)
                 .prefetch(runs_in_burst * 2)
                 )

        self.iterator = batch.make_initializable_iterator()
        it_tensors = self.iterator.get_next()

        # Assign all tensors to class variables
        self.true_x, self.time_x, self.true_y, self.time_y, self.vm_ix, self.norm_mean, self.norm_std = it_tensors

        self.encoder_features_depth = self.time_x.shape[2].value

    def load_vars(self, session):
        self.inp.restore(session)
        
    def init_iterator(self, session):
        session.run(self.iterator.initializer)

def train_features(inp: VarFeeder):
    return (inp.train_x, inp.train_y, inp.val_x, inp.val_y, inp.cities, inp.mean, inp.std, inp.month_autocorr, inp.week_autocorr)

def infer_features(inp: VarFeeder):
    return (inp.infer_x, inp.infer_y, inp.cities, inp.mean, inp.std, inp.month_autocorr, inp.week_autocorr)
