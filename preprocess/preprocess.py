import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from abc import *
from pathlib import Path
import os
import tempfile
import shutil
import pickle
import utils

#template
rating_score=0
user_core=5
item_core=5
# data_name='Apps_for_Android_5'
data_name='Movies_and_TV_5'
# data_name='Books_5'
# rating_score=0
# user_core=10
# item_core=20
# data_name='metadata'
#url='http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz'
#url='http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz'
url='http://snap.stanford.edu/data/amazon/productGraph/metadata.json.gz'
# utils.download_raw_dataset(url,data_name)
utils.preprocess(data_name,rating_score,user_core,item_core)
# utils.preprocess_bert(data_name,rating_score,user_core,item_core)