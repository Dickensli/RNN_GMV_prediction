import argparse
from hparams import build_from_set, build_hparams
from trainer import train
from make_features import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the model')
    
    # Prepare the data
    parser.add_argument('--city_path',
                        default='/nfs/isolation_project/intern/project/lihaocheng/city_forcast/city_day_features_to_yesterday.gbk.csv'
                        , help='Path that stores the city data')
    parser.add_argument('--weafor_path',
                        default='/nfs/isolation_project/intern/project/lihaocheng/city_forcast/weather_forecast.csv'
                        , help='Path that stores the weather forecast')
    parser.add_argument('--city_large', default=False, action='store_true', help='Whether to use large cities only')    
    parser.add_argument('--city_predict', default=False, action='store_true', help='Whether to predict 30 cities')    
    
    # Train the model
    parser.add_argument('--name', default='s32', help='Model name to identify different logs/checkpoints')
    parser.add_argument('--hparam_set', default='s32', help="Hyperparameters set to use (see hparams.py for available sets)")
    parser.add_argument('--n_models', default=3, type=int, help="Jointly train n models with different seeds")
    parser.add_argument('--multi_gpu', default=False,  action='store_true', help="Use multiple GPUs for multi-model training, one GPU per model")
    parser.add_argument('--seed', default=5, type=int, help="Random seed")
    parser.add_argument('--logdir', default='data/logs', help="Directory for summary logs")
    parser.add_argument('--datadir', default='data',
                        help="Directory to store the model/TF features/other temporary variables")
    parser.add_argument('--max_epoch', type=int, default=1000, help="Max number of epochs")
    parser.add_argument('--patience', type=int, default=100, help="Early stopping: stop after N epochs without improvement. Requires do_eval=True")
    parser.add_argument('--train_sampling', type=float, default=1.0, help="Sample this percent of data for training")
    parser.add_argument('--eval_sampling', type=float, default=1.0, help="Sample this percent of data for evaluation")
    parser.add_argument('--eval_memsize', type=int, default=5, help="Approximate amount of avalable memory on GPU, used for calculation of optimal evaluation batch size")
    parser.add_argument('--gpu', default=1, type=int, help='GPU instance to use')
    parser.add_argument('--gpu_allow_growth', default=False,  action='store_true', help='Allow to gradually increase GPU memory usage instead of grabbing all available memory at start')
    parser.add_argument('--save_best_model', default=True,  action='store_true', help='Save best model during training. Requires do_eval=True')
    parser.add_argument('--no_forward_split', default=True, dest='forward_split',  action='store_false', help='Use walk-forward split for model evaluation. Requires do_eval=True')
    parser.add_argument('--no_eval', default=True, dest='do_eval', action='store_false', help="Don't evaluate model quality during training")
    parser.add_argument('--no_summaries', default=True, dest='write_summaries', action='store_false', help="Don't Write Tensorflow summaries")
    parser.add_argument('--verbose', default=False, action='store_true', help='Print additional information during graph construction')
    parser.add_argument('--asgd_decay', type=float,  help="EMA decay for averaged SGD. Not use ASGD if not set")
    parser.add_argument('--no_tqdm', default=True, dest='tqdm', action='store_false', help="Don't use tqdm for status display during training")
    parser.add_argument('--max_steps', type=int, help="Stop training after max steps")
    parser.add_argument('--save_from_step', type=int, help="Save model on each evaluation (10 evals per epoch), starting from this step")
    parser.add_argument('--predict_window', default=288, type=int, help="Number of timestamps to predict")
    args = parser.parse_args()

    param_dict = dict(vars(args))
    param_dict['city_list'] = sorted(list(set(range(1, 357)) - {31, 181, 204, 205, 236, 237, 238, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 316}))
    if args.city_large:
        param_dict['city_list'] = sorted(list({1,  2,   3,   4,   5,   6,   7,   8,   9,  10,  12,  13,  14,
              15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  28, 29,
              32,  33,  34,  35,  36,  38,  39,  41,  44,  45,  46,  47,  48,
              50,  53,  58,  62,  63,  81,  82,  83,  84,  85,  86,  87,  88,
              89,  90,  92, 102, 105, 106, 118, 132, 133, 134, 135, 138, 142, 143,
              145, 153, 154, 157, 158, 159, 160, 173, 283} - {4, 11, 31}))
    if args.city_predict:
        param_dict['city_list'] = [2,3,5,6,7,9,10,15,16,21,22,23,25,26,28,29,32,34,35,36,38,39,41,50,53,63,105,118,134, 283]
    run(**param_dict)
    param_dict['hparams'] = build_from_set(args.hparam_set)
    del param_dict['hparam_set']
    train(**param_dict)
