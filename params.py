from re import A
from utils import fix_random_seed_as
import utils
import argparse
from template import set_template



parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42, help='random seed')

parser.add_argument('--template', type=str, default=None)

################
# Test
################
parser.add_argument('--test_model_path', type=str, default=None)

################
# Dataset
################
parser.add_argument('--dataset_name', type=str, default='book')
parser.add_argument('--rating_score', type=int, default=4, help='Only keep ratings greater than equal to this value')
parser.add_argument('--min_uc', type=int, default=5, help='Only keep users with more than min_uc ratings')
parser.add_argument('--min_sc', type=int, default=0, help='Only keep items with more than min_sc ratings')
parser.add_argument('--max_len',type=int,default=50,help='The max length of items sequence')
################
# Dataloader
################
parser.add_argument('--dataloader_random_seed', type=float, default=0.0)
parser.add_argument('--train_batch_size', type=int, default=8)
parser.add_argument('--val_batch_size', type=int, default=8)
parser.add_argument('--test_batch_size', type=int, default=8)
parser.add_argument('--bert_mask_prob', type=float, default=None, help='Probability for masking items in the training sequence')
################
# NegativeSampler
################
parser.add_argument('--train_negative_sampler_code', type=str, default='random', choices=['popular', 'random'],
        help='Method to sample negative items for training. Not used in bert')
parser.add_argument('--train_negative_sample_size', type=int, default=100)
parser.add_argument('--train_negative_sampling_seed', type=int, default=None)
parser.add_argument('--test_negative_sampler_code', type=str, default='random', choices=['popular', 'random'],
        help='Method to sample negative items for evaluation')
parser.add_argument('--test_negative_sample_size', type=int, default=100)
parser.add_argument('--test_negative_sampling_seed', type=int, default=None)

################
# Trainer
parser.add_argument("--local_rank", type=int, help="number of cpu threads to use during batch generation")
################
# device #
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--device_idx', type=str, default='0')
# optimizer #
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--eps',type=float,default=1e-8)
# lr scheduler #
parser.add_argument('--decay_step', type=int, default=15, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
# epochs #
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
#logger#
parser.add_argument('--log_period_as_iter', type=int, default=12800)
# evaluation #
parser.add_argument('--metric_ks', nargs='+', type=int, default=[10, 20, 50], help='ks for Metric@k')
parser.add_argument('--best_metric', type=str, default='NDCG@10', help='Metric for determining the best model')

################
# Experiment
################
parser.add_argument('--experiment_dir', type=str, default='experiments')
parser.add_argument('--experiment_description', type=str, default='test')


args=parser.parse_args()    
# Set seeds
args=set_template(args)  

 