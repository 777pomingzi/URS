import imp
from torch.utils.data import DataLoader,Dataset,Sampler
from pathlib import Path
from collections import defaultdict
import json
import gzip
import random
from tqdm import tqdm
import torch
import numpy as np
import os
import pickle
from negative_samplers import negative_sampler_factory



def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

class Amazon_Dataset(Dataset):
    def __init__(self,args,mode='train'):
        self.mode=mode
        self.max_len=args.max_len
        self.template=args.template
        data_path=Path('preprocess')
        folder_name='{}_min_rating{}-min_uc{}-min_sc{}'\
        .format(args.dataset_name,args.rating_score,args.min_uc,args.min_sc)

        save_folder=data_path.joinpath('preprocessed',folder_name)

        data_path=data_path.joinpath('preprocessed',folder_name,'dataset.pkl')

        
        dataset=load_pickle(data_path)
        if self.mode =='train':
            self.review_data=dataset['train']
        elif self.mode =='val':
            self.review_data=dataset['val']
        elif self.mode =='test':
            self.review_data=dataset['test']
        else:
            raise NotImplementedError

        
        self.umap = dataset['umap']
        self.smap = dataset['smap']

        self.users=list(self.umap.keys())
        self.items=list(self.smap.keys())

        code = args.test_negative_sampler_code
        self.user_count=len(self.umap)
        self.item_count=len(self.smap)

        test_negative_sampler = negative_sampler_factory(code, dataset['test'],
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         save_folder,self.users,self.items)
        # if self.mode !='train':
        self.test_negative_samples = test_negative_sampler.get_negative_samples()


    def __len__(self):
        return len(self.review_data)


    def __getitem__(self,index):
        #print(index)
        #print(len(self.users))
        user=self.users[index]
        #print(user)
        user_history=self.review_data[user]
        #print(len(user_history))
        #print(user_history)
        seq=user_history[-(self.max_len+1):-1]
        ans=user_history[-1:]
        #print(ans)
        history=''
        his_len=len(seq)-1
        for i,item in enumerate(seq):
            history+=item
            if i!=his_len:
                history+=' , '
            else:
                history+=' .'
        # source_txt=self.template['source'].format('',history)
        # source_txt=self.template['source'].format(user,history)
        source_txt=self.template['source'].format(history)
        target_text = self.template['target'].format(ans)
        

        if self.mode=='train':
            return source_txt,target_text
        else:
            return source_txt,target_text,self.test_negative_samples[user]
            
        
            
        