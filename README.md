### Env
1. tensorflow-1.4.1
2. python-3.6 / python3.5
3. pandas
4. numba, argsparse, tqdm etc.

### Data fetch
Main data path: `'/nfs/isolation_project/intern/project/lihaocheng/city_forcast/city_day_features_to_yesterday.gbk.csv'`  
                `'/nfs/isolation_project/intern/project/lihaocheng/city_forcast/weather_forecast.csv'` -> pd.DataFrame

### Main files
> generate_model.py - main file to do data prepossessing and tarin the model. Dependency: make_features.py, model.py, trainer.py  
> feature_server.py - script to generate tarin/val/test features and store them into pandas dataFrames
> make_features.py - builds features from source data and store in TF tensor  
> input_pipe.py - TF data preprocessing pipeline (assembles features into training/evaluation tensors, performs some sampling and normalisation)  
> model.py - the model  
> trainer.py - trains the model(s)  
> hparams.py - hyperpatameter sets.   
> predict.py - generate predictions and csv for each vm with format 'prediction/true_value' vs. timestamps  

### Execute
Choose a folder to store the weight/tensor/result information. In this demo we create `data`.

Run `python generate_model.py --city_large --gpu=0 --name=s32 --hparam_set=s32 --n_model=3 \
-asgd_decay=0.99 --logdir=data/logs --datadir=data --seed=5 \
--city_path=/nfs/isolation_project/intern/project/lihaocheng/city_forcast/city_day_features_to_yesterday.gbk.csv \  
--weafor_path=/nfs/isolation_project/intern/project/lihaocheng/city_forcast/weather_forecast.csv`  
to do:  
1. extract features:     
    The features will be crafted in `feature_server.py` and passed to `make_features.py` to generate TF tensor. We utilize `rain_hour_cont`, `intensity` and `weekday` info and make cross features and non-linearity terms. The result will be stored in **datadir**`/vars` (`data/vars`) as Tensorflow checkpoint. There are two options `--city_large` or `--city_predict` which use 75 or 30 cities for training.

2. Train model:  
    Simultaneously train **n_model** models based on different seeds (on a single TF graph). Hyperparameters are described as **s32** from `hparam.py`.  
    Exponential moving average is adjustable with default value **asgd_decay**. Predict window is fixed to **predict_window** fot both training and prediction stages.  
    Log history will be stored in **logdir**. Multiple model generated during the training will be stored in **datadir**`/cpt/s32` (`data/cpt/s32`) and we will pick the best one for prediction.  
  
Run `python predict.py --city_large --weight_path=data/cpt/s32 --datadir=data --n_models=3 --seed=5` to make a prediction and also I provide `--city_large` and `--city_predict` two options.
