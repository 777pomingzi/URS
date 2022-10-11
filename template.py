#设定模型的input和target
import torch
template={}
template['source'] = "Given the following purchase history of user_{} : \n {} \n predict masked item to be purchased by the user ?"
# template['source'] = "Given the following purchase history of user_{} : \n {} \n predict next possible item to be purchased by the user ?"
# template['source']="{}"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_name', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_name']

def set_template(args):
        print(torch.cuda.device_count())  
        torch.distributed.init_process_group(backend="nccl",init_method='env://')
    #if args.template is None:
        #return 
    #else:
        args.template=template
        # args.dataset_name='Movies_and_TV_Books_5'
        args.dataset_name='Books_5'
        # args.dataset_name='Movies_and_TV_5'
        args.rating_score=0
        args.min_uc=10
        args.min_sc=10
        # args.min_uc=5
        # args.min_sc=5
        args.max_len=50
        batch=4
        args.train_batch_size=batch
        args.val_batch_size=batch
        args.test_batch_size=batch
        args.bert_mask_prob=0.2

        args.train_negative_sampler_code='random'
        args.train_negative_sample_size=0
        args.train_negative_sampling_seed=0
        args.test_negative_sampler_code='random'
        args.test_negative_sample_size=100
        args.test_negative_sampling_seed=98765

        args.local_rank=torch.distributed.get_rank()
        # print(args.local_rank)
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        # print(args.device)
        # args.device='cuda'
        args.num_gpu=1
        # args.device_idx='0'

        args.num_beams=20
        args.lr=0.00001
        args.decay_step=25
        args.gamma=1.0
        args.num_epochs=20
        args.metric_ks=[1,5,10,20,50]
        args.best_metric='NDCG@10'
        args.seed=42
        # args.seed=680
        return args