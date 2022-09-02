import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import pprint as pp
import os
from datetime import date
from pathlib import Path
import pickle

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic=True
    cudnn.benchmark=False


class AverageMeterSet(object):
    def __init__(self,meters=None):
        self.meters=meters if meters else{}
    
    def __getitem__(self,key):
        if key not in self.meters:
            meter=AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self,name,value,n=1):
        if name not in self.meters:
            self.meters[name]=AverageMeter()
        self.meters[name].update(value,n)

    def reset(self):
        for meter in self.meters:
            meter.reset()
    
    def values(self,format_string='{}'):
        return {format_string.format(name):meter.val for name,meter in self.meters.items()}

    def averages(self,format_string='{}'):
        return {format_string.format(name):meter.avg for name,meter in self.meters.items()}

    def sums(self,format_string='{}'):
        return {format_string.format(name):meter.sum for name,meter in self.meters.items()}

    def counts(self,format_string='{}'):
        return {format_string.format(name):meter.count for name,meter in self.meters.items()}


class AverageMeter(object):

    def __init__(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0

    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0
    
    def update(self,val,n=1):
        self.val=val
        self.sum+=val
        self.count+=n
        self.avg=self.sum/self.count

        
    def __format__(self,format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self,format=format)


def setup_train(args):

    export_root=create_experiment_export_folder(args)
    if args.local_rank==0:
        pp.pprint({k:v for k,v in vars(args).items() if v is not None},width=1)
    return export_root

def create_experiment_export_folder(args):
    experiment_dir,experiment_description=args.experiment_dir,args.experiment_description
    if not os.path.exists(experiment_dir) and args.local_rank==0 :
        os.makedirs(experiment_dir)
    experiment_path=get_name_of_experiment_path(experiment_dir,experiment_description)
    if args.local_rank==0 and not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
        print('Folder created: '+os.path.abspath(experiment_path))
    return experiment_path


def get_experiment_index(experiment_path):
    idx = 0 
    while os.path.exists(experiment_path+"_"+str(idx)):
        idx+=1
    return idx

    
def get_name_of_experiment_path(experiment_dir,experiment_description):
    experiment_path=os.path.join(experiment_dir,(experiment_description+"_"+str(date.today())))
    idx=get_experiment_index(experiment_path)
    experiment_path=experiment_path+"_"+str(idx)
    return experiment_path


def get_items(args):
    data_path=Path('preprocess')
    folder_name='{}_min_rating{}-min_uc{}-min_sc{}'\
    .format(args.dataset_name,args.rating_score,args.min_uc,args.min_sc)
    data_path=data_path.joinpath('preprocessed',folder_name,'dataset.pkl')
    items=load_pickle(data_path)['smap'].keys()
    return items

def recall(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / labels.sum(1).float()).mean().item()


def ndcg(scores, labels, k):
    # scores = scores.cpu()
    # labels = labels.cpu()
    rank = (scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(n, k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}

    scores = scores.cpu()
    labels = labels.cpu()
    answer_count = labels.sum(1)
    answer_count_float = answer_count.float()
    labels_float = labels.float()
    rank = (scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
       cut = cut[:, :k]
    #    print(cut)
       hits = labels_float.gather(1, cut)
       metrics['Recall@%d' % k] = (hits.sum(1) / answer_count_float).mean().item()

       position = torch.arange(2, 2+k)
       weights = 1 / torch.log2(position.float())
       dcg = (hits * weights).sum(1)
       idcg = torch.Tensor([weights[:min(n, k)].sum() for n in answer_count])
       ndcg = (dcg / idcg).mean()
       metrics['NDCG@%d' % k] = ndcg
    pos=[]
    for i in range(len(scores)):
        for _ in range(3):
            pos.append(i)
    
    row_index=np.array(pos)
    col_index=rank[:,:3].reshape(-1)

    # return metrics
    return metrics,row_index,col_index

