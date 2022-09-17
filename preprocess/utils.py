from genericpath import isfile
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


from pathlib import Path
import os

import pickle
import tempfile
import shutil
import zipfile
import wget
import gzip


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def preprocess(dataset_name,rating_score,min_uc,min_sc):
    data_path=Path('preprocessed')
    folder_name='{}_min_rating{}-min_uc{}-min_sc{}'\
        .format(dataset_name,rating_score,min_uc,min_sc)
    data_path=data_path.joinpath(folder_name,'dataset.pkl')
    if data_path.is_file():
        print('Already preprocessed. Skip preprocessing')
        return 
    if not data_path.parent.is_dir():
        data_path.parent.mkdir(parents=True)
    
    df_review=Amazon(dataset_name,rating_score)
    # df_review=filter_na(df_review)
    # print(len(df_review))
    df_meta=Amazon_meta()    

    
    
    df_review=filter_triplets(df_review,min_sc,min_uc)
    print(len(df_review))
    df_m=df_merge(df_review,df_meta)
    print(len(df_m))
    print(df_m)
    df_m=filter_na(df_m)
    print(len(df_m))
    print(df_m)
    df_m,umap,smap=densify_index(df_m)
    print(umap)
    train,val,test,umap=split_df(df_m,umap,min_uc)
    dataset={'train':train,
                'val':val,
                'test':test,
                'umap':umap,
                'smap':smap}
    with data_path.open('wb') as f:
        pickle.dump(dataset,f)    



def Amazon(dataset_name, rating_score):
    '''
    reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
    asin - ID of the product, e.g. 0000013714
    reviewerName - name of the reviewer
    helpful - helpfulness rating of the review, e.g. 2/3
    --"helpful": [2, 3],
    reviewText - text of the review
    --"reviewText": "I bought this for my husband who plays the piano. ..."
    overall - rating of the product
    --"overall": 5.0,
    summary - summary of the review
    --"summary": "Heavenly Highway Hymns",
    unixReviewTime - time of the review (unix time)
    --"unixReviewTime": 1252800000,
    reviewTime - time of the review (raw)
    --"reviewTime": "09 13, 2009"
    '''

    file_path = '../Data/reviews_' + dataset_name + '.json/'+dataset_name+'.json'
    fin=open(file_path,'r')

    df={}
    useless_col=['helpful','reviewText','summary','reviewTime']
    i=0
    for line in fin:
        d=eval(line)
        if float(d['overall'])<rating_score:
            continue
        for s in useless_col:
            if s in d:
                d.pop(s)
        df[i]=d
        i+=1
    return pd.DataFrame.from_dict(df,orient='index')



def Amazon_meta():
    #meta数据集内容
    file_path='../Data/metadata.json/metadata.json'
    fin_meta =open(file_path,'r')

    df_meta={}
    useless_col = ['price','imUrl','related','salesRank','description','brand','categories'] 
    i=0
    for line in fin_meta:
        d=eval(line)
        for s in useless_col:
            if s in d:
                d.pop(s)
        df_meta[i]=d
        i+=1
    return pd.DataFrame.from_dict(df_meta,orient='index')


#过滤交互次数较少的user和item
def filter_triplets(df,min_sc,min_uc):
        print('Filtering triplets')
        if min_sc>0:
            item_sizes=df.groupby('asin').size()
            good_items=item_sizes.index[item_sizes>=min_sc]
            df=df[df['asin'].isin(good_items)]
        if min_uc>0:
            user_sizes=df.groupby('reviewerID').size()
            good_users=user_sizes.index[user_sizes>min_uc]
            df=df[df['reviewerID'].isin(good_users)]
        return df


def densify_index(df):
    print('Densifying index')
    # umap={u:(i+1) for i,u in enumerate(set(df['reviewerID']))}
    umap={u:(i+1) for i,u in enumerate(sorted(list(set(df['reviewerID']))))}
    smap={s:(i+1) for i,s in enumerate(set(df['title']))}
    print(len(umap))
    print("--------------")
    print(len(smap))
    df['reviewerID']=df['reviewerID'].map(umap)
    #df['title']=df['title'].map(smap)
    return df,umap,smap


def df_merge(left,right):
    return pd.merge(left, right, on=['asin'],how='left')


def split_df(df,umap,min_uc):
    user_group=df.groupby('reviewerID')
    print(len(user_group))
    user_seq=user_group.progress_apply(lambda d:list(d.sort_values(by='unixReviewTime')['title']))
    train,val,test={},{},{}
    num=0
    sum=0
    # for user in umap:
        # query="reviewerID=='"+user+"'"
        # username=df.query(query).head(1)['reviewerName'].item()
        # items=user_seq[user]
        # if(len(items)>=min_uc) and username not in train.keys(): 
        #     num+=1
        #     sum+=len(items)
        #     train[username],val[username],test[username]=items[:-2],items[:-1],items[:]
    user_pool=set()
    for i in range(1,len(umap)+1):
        # user_id=map_u[i]
        query="reviewerID=="+str(i)
        username=df.query(query).head(1)['reviewerName'].item()
        print(username)
        if username in user_pool:
            continue
        user_pool.add(username)
        items=user_seq[i]
        if(len(items)>=min_uc):
        # if(len(items)>=min_uc) and username not in train.keys():
            train[username],val[username],test[username]=items[:-2],items[:-1],items[:]
            num+=1
            sum+=len(items)
    print('num of user sequence : ',num)
    print('average length of user sequence : ',sum/num)
    users=list(train.keys())
    umap={u:i for i,u in enumerate(users)}
    return train,val,test,umap



def download_raw_dataset(url,name,is_zipfile=True):
    folder_path='../Data'
    print("Raw file doesn't exist. Downloading...")
    if is_zipfile:
        tmproot = Path(tempfile.mkdtemp())
        tmpzip = tmproot.joinpath('file.gz')
        tmpfolder = tmproot.joinpath(name+'.json')
        download(url, tmpzip)
        unzip(tmpzip, tmpfolder)
        shutil.move(tmpfolder, folder_path)
        shutil.rmtree(tmproot)
        print()


def download(url, savepath):
    wget.download(url, str(savepath))


def unzip(zippath, savepath):
    zippath=str(zippath)
    f_name = zippath.replace(".gz","")
    # 开始解压
    g_file = gzip.GzipFile(zippath)
    #读取解压后的文件，并写入去掉后缀名的同名文件（即得到解压后的文件）
    open(f_name, "wb+").write(g_file.read())
    g_file.close()
    shutil.move(f_name,savepath)


def filter_na(df):
    df=df.dropna()
    return df


    
def preprocess_bert(dataset_name,rating_score,min_uc,min_sc):
    data_path=Path('preprocessed_bert')
    folder_name='{}_min_rating{}-min_uc{}-min_sc{}'\
        .format(dataset_name,rating_score,min_uc,min_sc)
    data_path=data_path.joinpath(folder_name,'dataset.pkl')
    if data_path.is_file():
        print('Already preprocessed. Skip preprocessing')
        return 
    if not data_path.parent.is_dir():
        data_path.parent.mkdir(parents=True)

    df_review=Amazon(dataset_name,rating_score)
    df_meta=Amazon_meta()
    
    df_review=filter_triplets(df_review,min_sc,min_uc)
    df_m=df_merge(df_review,df_meta)
    print(df_m)
    df_m=filter_na(df_m)
    print(df_m)
    df_m,umap,smap=densify_index_bert(df_m)
    print(umap)
    train,val,test,umap=split_df_bert(df_m,umap,min_uc)
    print(len(umap))
    dataset={'train':train,
                'val':val,
                'test':test,
                'umap':umap,
                'smap':smap}
    with data_path.open('wb') as f:
        pickle.dump(dataset,f)   



def Amazon_bert(dataset_name, rating_score):
    '''
    reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
    asin - ID of the product, e.g. 0000013714
    reviewerName - name of the reviewer
    helpful - helpfulness rating of the review, e.g. 2/3
    --"helpful": [2, 3],
    reviewText - text of the review
    --"reviewText": "I bought this for my husband who plays the piano. ..."
    overall - rating of the product
    --"overall": 5.0,
    summary - summary of the review
    --"summary": "Heavenly Highway Hymns",
    unixReviewTime - time of the review (unix time)
    --"unixReviewTime": 1252800000,
    reviewTime - time of the review (raw)
    --"reviewTime": "09 13, 2009"
    '''

    file_path = '../Data/reviews_' + dataset_name + '.json/'+dataset_name+'.json'
    fin=open(file_path,'r')

    df={}
    useless_col=['helpful','reviewText','summary','reviewTime']
    i=0
    for line in fin:
        d=eval(line)
        if float(d['overall'])<rating_score:
            continue
        for s in useless_col:
            if s in d:
                d.pop(s)
        df[i]=d
        i+=1
    return pd.DataFrame.from_dict(df,orient='index')    








def densify_index_bert(df):
    print('Densifying index')
    # umap={u:(i+1) for i,u in enumerate(set(df['reviewerID']))}
    umap={u:(i+1) for i,u in enumerate(sorted(list(set(df['reviewerID']))))}
    smap={s:(i+1) for i,s in enumerate(set(df['asin']))}
    print(len(umap))
    print("--------------")
    print(len(smap))
    df['reviewerID']=df['reviewerID'].map(umap)
    df['asin']=df['asin'].map(smap)
    return df,umap,smap



def split_df_bert(df,umap,min_uc):
    print('split_df_bert')
    user_group=df.groupby('reviewerID')
    print(len(user_group))
    user_seq=user_group.progress_apply(lambda d:list(d.sort_values(by='unixReviewTime')['asin']))
    train,val,test={},{},{}
    user=0
    user_pool=set()
    # map_u={(i+1):u for i,u in enumerate(umap)}
    for i in range(1,len(umap)+1):
        # user_id=map_u[i]
        query="reviewerID=="+str(i)
        username=df.query(query).head(1)['reviewerName'].item()
        print(username)
        if username in user_pool:
            continue
        user_pool.add(username)
        items=user_seq[i]
        if(len(items)>=min_uc):
            train[user],val[user],test[user]=items[:-2],items[-2:-1],items[-1:]
            user+=1
    users=list(train.keys())
    # df_user=df['reviewerID']
    umap={u:(i+1) for i,u in enumerate(users)}
    return train,val,test,umap
