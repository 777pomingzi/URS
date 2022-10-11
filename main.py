from Dataset import Amazon_dataset
from train import trainer
from torch.utils.data import DataLoader,DistributedSampler
from pathlib import Path
from params import args
from utils import *
def train(args):
    fix_random_seed_as(args.seed)
    train_dataset=Amazon_dataset.Amazon_Dataset(args,'train')
    val_dataset=Amazon_dataset.Amazon_Dataset(args,'val')
    test_dataset=Amazon_dataset.Amazon_Dataset(args,'test')
    # for i in range (10):
    #     _,_,negs=val_dataset[i]
    #     print(len(negs))
    #     print('------------------------')
    # input,label=train_dataset[2]
    # print('------------------------------------')
    # print(input)
    # print(label)
    # for i in range(10):
    #     print(train_dataset[i])
    # print(val_dataset[10])
    # print(test_dataset[10])
    train_sampler=DistributedSampler(train_dataset)
    val_sampler=DistributedSampler(val_dataset)
    test_sampler=DistributedSampler(test_dataset)

    # items=get_items(args)
    train_dataloader= DataLoader(train_dataset,batch_size=args.train_batch_size, pin_memory=True,sampler=train_sampler)   
    val_dataloader= DataLoader(val_dataset,batch_size=args.val_batch_size, pin_memory=True,sampler=val_sampler)
    test_dataloader= DataLoader(test_dataset,batch_size=args.test_batch_size, pin_memory=True,sampler=test_sampler)
    
    Trainer=trainer.Amazon_trainer(args,setup_train(args),train_dataloader,val_dataloader,test_dataloader)
    Trainer.train()

if __name__=='__main__':
    train(args)