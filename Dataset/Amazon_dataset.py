from email.contentmanager import raw_data_manager
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
        self.args=args
        self.mode=mode
        self.rng=random.Random(args.dataloader_random_seed)
        self.max_len=args.max_len
        self.template=args.template
        self.mask_prob = args.bert_mask_prob
        data_path=Path('preprocess')
        folder_name='{}_min_rating{}-min_uc{}-min_sc{}'\
        .format(args.dataset_name,args.rating_score,args.min_uc,args.min_sc)
        
        # if self.mode=='train':
        #     folder_name='Books_5_min_rating0-min_uc10-min_sc10'
        save_folder=data_path.joinpath('preprocessed',folder_name)

        data_path=data_path.joinpath('preprocessed',folder_name,'dataset.pkl')

        
        dataset=load_pickle(data_path)
        if self.mode =='train':
            # raw_data=dataset['train']
            # self.review_data={}
            # for k in raw_data:
            #     if len(raw_data[k])>15:# 无短序
            #         self.review_data[k]=raw_data[k]
            # for k in raw_data:
            #     if len(raw_data[k])<=15 or len(raw_data[k])>30:# 无中序
            #         self.review_data[k]=raw_data[k]        
            # for k in raw_data:
            #     if len(raw_data[k])<=30:# 无长序
            #         self.review_data[k]=raw_data[k]
            self.review_data=dataset['train']
        elif self.mode =='val':
            self.review_data=dataset['val']
        elif self.mode =='test':
            self.review_data=dataset['test']
        else:
            raise NotImplementedError

        # if self.mode =='train':
        #     self.review_data=dataset['train']
        # elif self.mode =='val':
        #     raw_data=dataset['val']
        #     self.review_data={}
        #     # for k in raw_data:
        #     #     if len(raw_data[k])<=15:# 评判短序
        #     #         self.review_data[k]=raw_data[k]
        #     # for k in raw_data:
        #     #     if len(raw_data[k])>=15 and len(raw_data[k])<=30:# 评判中序
        #     #         self.review_data[k]=raw_data[k]        
        #     for k in raw_data:
        #         if len(raw_data[k])>=30:# 评判长序
        #             self.review_data[k]=raw_data[k]
        # elif self.mode =='test':
        #     raw_data=dataset['test']
        #     self.review_data={}
        #     # for k in raw_data:
        #     #     if len(raw_data[k])<=15:# 评判短序
        #     #         self.review_data[k]=raw_data[k]
        #     # for k in raw_data:
        #     #     if len(raw_data[k])>=15 and len(raw_data[k])<=30:# 评判中序
        #     #         self.review_data[k]=raw_data[k]        
        #     for k in raw_data:
        #         if len(raw_data[k])>=30:# 评判长序
        #             self.review_data[k]=raw_data[k]
        # else:
        #     raise NotImplementedError

        
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.users=list(self.review_data.keys())
        # self.users=list(self.umap.keys())
        self.items=list(self.smap.keys())

        code = args.test_negative_sampler_code
        self.user_count=len(self.users)
        self.item_count=len(self.smap)

        if self.mode !='train':
            test_negative_sampler = negative_sampler_factory(code, dataset['test'],
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         save_folder,self.users,self.items)
            self.test_negative_samples = test_negative_sampler.get_negative_samples()


    def __len__(self):
        return len(self.review_data)


    def __getitem__(self,index):
        # print(index)
        #print(len(self.users))
        user=self.users[index]
        #print(user)
        user_history=self.review_data[user]
        #print(len(user_history))
        #print(user_history)
        if self.mode=='train':
            history_sample=self.rng.randint(self.args.min_uc-2,len(user_history))
            user_history=user_history[:history_sample]
        seq=user_history[-self.max_len:-1]
        ans=user_history[-1:]
        ans=ans[0]
        #print(ans)
        history=''
        his_len=len(seq)-1
        for i,item in enumerate(seq):
            history+=item
            if i!=his_len:
                history+=' , '
            else:
                history+=' .'
        # if self.mode=='train':
        #     user=self.users[index]
        #     user_history=self.review_data[user][-self.max_len:]
        #     id=0
        #     inputs=[]
        #     labels=[]
        #     for item in user_history:
        #         prob = self.rng.random()
        #         if prob < self.mask_prob:
        #             prob /= self.mask_prob

        #             if prob < 0.8:
        #                 sentinel='<extra_id_{}>'
        #                 sentinel=sentinel.format(id)
        #                 id+=1
        #                 inputs.append(sentinel)
        #                 sentinel=sentinel+' '+item
        #                 labels.append(sentinel)

        #             elif prob < 0.9:
        #                 inputs.append(self.items[self.rng.randint(0,self.item_count-1)])
        #             else:
        #                 inputs.append(item)
        #         else:
        #             inputs.append(item)
        #     history=''
        #     his_len=len(inputs)-1
        #     for i,item in enumerate(inputs):
        #         history+=item
        #         if i!=his_len:
        #             history+=' , '
        #         else:
        #             history+=' .'
        #     ans=''
        #     ans_len=len(labels)-1
        #     for i,item in enumerate(labels):
        #         ans+=item
        #         if i != ans_len:
        #             ans+=' '
        # else:
        #     #print(len(self.users))
        #     user=self.users[index]
        #     #print(user)
        #     user_history=self.review_data[user]
        #     #print(len(user_history))
        #     #print(user_history)
        #     seq=user_history[-self.max_len:-1]
        #     ans=user_history[-1:]
        #     ans=ans[0]
        #     #print(ans)
        #     history=''
        #     his_len=len(seq)-1
        #     for i,item in enumerate(seq):
        #         history+=item
        #         if i!=his_len:
        #             history+=' , '
        #         else:
        #             # history+=' .' 
        #             history+=' , <extra_id_0>.' 
        #     ans='<extra_id_0> '+ans         
        # source_txt=self.template['source'].format('',history)
        source_txt=self.template['source'].format(user,history)
        # source_txt=self.template['source'].format(history)
        target_text = self.template['target'].format(ans)
        

        if self.mode=='train':
            return source_txt,target_text
        else:
            return source_txt,target_text,self.test_negative_samples[user]
            
        
            
        