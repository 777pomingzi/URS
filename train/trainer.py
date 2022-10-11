from wsgiref import validate
from transformers import  T5Tokenizer,T5ForConditionalGeneration,AdamW,BartTokenizer, BartModel,T5Config
from abc import ABCMeta, abstractmethod
from Dataset import Amazon_dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from torch import nn
from utils import AverageMeterSet, recalls_and_ndcgs_for_ks
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
import numpy as np
import os
import random
class Amazon_trainer(object):
    def __init__(self,args,export_root,train_loader=None, val_loader=None, test_loader=None):
        self.args = args
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.test_loader=test_loader
        self.device=args.device
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        self.rng=random.Random(self.args.seed)
        # self.tokenizer = T5Tokenizer.from_pretrained('t5-large')
        # self.model = T5ForConditionalGeneration.from_pretrained('t5-large').to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base').to(self.device)
        # config = T5Config().from_pretrained('t5-base')
        # self.model = T5ForConditionalGeneration(config).to(self.device)

        # self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        # self.model = BartModel.from_pretrained('facebook/bart-base').to(self.device)
        #self.prefix_trie=self.construct_trie(items)
        #self.prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist())

        self.num_epochs=args.num_epochs
        
        self.export_root=export_root
        self.log_period_as_iter=args.log_period_as_iter

        self.is_parallel = args.num_gpu > 1
        if self.is_parallel:
            self.model = nn.parallel.DistributedDataParallel(self.model,device_ids=[args.local_rank],
                                                      output_device=args.local_rank)
            print('The num of gpu is : ',torch.cuda.device_count())

        self.writer, self.train_loggers,self.val_loggers= self._create_loggers()
        self.logger_service = LoggerService(self.train_loggers,self.val_loggers)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
        
            },
        ]

        self.optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.lr, eps=args.eps)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_step, gamma=args.gamma)


    def train(self):
        self.model.train()
        accum_iter = 0
        # self.validate(0,accum_iter) 
        for epoch in range(self.num_epochs):

            average_meter_set = AverageMeterSet()
            tqdm_dataloader=tqdm(self.train_loader)

            for input,label in tqdm_dataloader:\
                #sliding-window用于随机采样
                # dp_prob=self.rng.random()
                # if dp_prob<0.95:
                #     continue
                batch_size = len(label)

                self.optimizer.zero_grad()

                tokenized_input=self.tokenizer(input,max_length=512,padding='max_length',return_tensors="pt",truncation=True)
                tokenized_label=self.tokenizer(label,max_length=32,padding='max_length',return_tensors="pt",truncation=True)

                input_ids=tokenized_input["input_ids"].to(self.device)
                attention_mask=tokenized_input["attention_mask"].to(self.device)

                labels=tokenized_label["input_ids"].to(self.device)
                decoder_attention_mask=tokenized_label["attention_mask"].to(self.device)

                labels[labels == self.tokenizer.pad_token_id] = -100
                
                output=self.model(input_ids=input_ids,labels=labels,attention_mask=attention_mask,decoder_attention_mask=decoder_attention_mask)

                loss=output["loss"]
                # loss=loss.mean()
                
                loss.backward()
                self.optimizer.step()
                
                average_meter_set.update('loss',loss.item())
                tqdm_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch+1, average_meter_set['loss'].avg))

                accum_iter += batch_size

                if self._needs_to_log(accum_iter):
                    log_data = {
                        'state_dict': (self._create_state_dict()),
                        'epoch': epoch,
                        'accum_iter': accum_iter,
                    }
                    log_data.update(average_meter_set.averages())
                    self.logger_service.log_train(log_data)
            # if epoch%5==0:
            # if epoch>5:
            self.validate(epoch,accum_iter) 
        # self.validate(epoch,accum_iter) 
        self.test(accum_iter)
        self.writer.close()
           

    def validate(self,epoch,accum_iter):#tokenizer输入(B)
        self.model.eval()
        
        with torch.no_grad():
            average_meter_set=AverageMeterSet()
            #测试在训练数据集上有效，记得改回去
            tqdm_dataloader=tqdm(self.val_loader)
            # tqdm_dataloader=tqdm(self.train_loader)
            for inputs,labels,negs in tqdm_dataloader:#negs:(N,B)
                num_negs=len(negs)
                negs=np.array(negs).T#B*N
                negs=negs.tolist()
                
                tokenized_inputs=self.tokenizer(inputs,max_length=512,padding='max_length',return_tensors="pt",truncation=True)
                input_ids=tokenized_inputs["input_ids"].to(self.device)#B*T
                # print(input_ids.shape)
                attention_mask=tokenized_inputs["attention_mask"].to(self.device)#B*T

                tokenized_labels=self.tokenizer(labels,max_length=32,padding='max_length',return_tensors="pt",truncation=True)#B*T
                label_ids=tokenized_labels["input_ids"].to(self.device)
                decoder_attention_mask=tokenized_labels["attention_mask"].to(self.device)
                label_ids[label_ids == self.tokenizer.pad_token_id] = -100
                

                outputs=self.model(input_ids=input_ids,attention_mask=attention_mask,decoder_attention_mask=decoder_attention_mask , labels=label_ids,output_hidden_states=True,output_attentions=True)
                
                encoder_last_hidden_state=outputs['encoder_last_hidden_state']#B*T*H
                encoder_last_hidden_state=encoder_last_hidden_state.unsqueeze(1).expand(-1,num_negs+1,-1,-1)

                attention_masks=attention_mask.unsqueeze(1).expand(-1,num_negs+1,-1)

                neg_ids=None
                for neg in negs:#neg(N)
                    tokenized_neg=self.tokenizer(neg,max_length=32,padding='max_length',return_tensors="pt",truncation=True)#(N,T)
                    neg_id=tokenized_neg["input_ids"].to(self.device)
                    neg_decoder_attention_mask=tokenized_neg["attention_mask"].to(self.device)
                    neg_id[neg_id == self.tokenizer.pad_token_id] = -100
                    neg_id=neg_id.unsqueeze(0)
                    neg_decoder_attention_mask=neg_decoder_attention_mask.unsqueeze(0)
                    if neg_ids is None:
                        neg_ids=neg_id
                        neg_decoder_attention_masks=neg_decoder_attention_mask
                    else:
                        neg_ids=torch.cat((neg_ids,neg_id),dim=0)
                        neg_decoder_attention_masks=torch.cat((neg_decoder_attention_masks,neg_decoder_attention_mask),dim=0)

                label_ids=label_ids.unsqueeze(1)#(B,1,T)
                decoder_attention_mask=decoder_attention_mask.unsqueeze(1)    
                label_ids=torch.cat((label_ids,neg_ids),dim=1)#(B,(N+1),T)
                decoder_attention_mask=torch.cat((decoder_attention_mask,neg_decoder_attention_masks),dim=1)

                
                scores_mark=None
                for attention_mask,hidden_state,candidate_ids,decoder_attention_masks in zip(attention_masks,encoder_last_hidden_state,label_ids,decoder_attention_mask):#(N+1,T,H) (N+1,T)
                    # print(hidden_state.shape)
                    # print(candidate_ids.shape)#((N+1),T)
                    
                    # logits=self.model(attention_mask=attention_mask[:51],encoder_outputs=(hidden_state[:51],None,None),decoder_attention_mask=decoder_attention_masks[:51] , labels=candidate_ids[:51])['logits']
                    # logits=torch.cat((logits,self.model(attention_mask=attention_mask[51:],encoder_outputs=(hidden_state[51:],None,None),decoder_attention_mask=decoder_attention_masks[51:] , labels=candidate_ids[51:])['logits']),dim=0)
                    logits=self.model(attention_mask=attention_mask[:51],encoder_outputs=(hidden_state[:51],None,None),decoder_attention_mask=decoder_attention_masks[:51] , labels=candidate_ids[:51])['logits']
                    # logits=torch.cat((logits,self.model(attention_mask=attention_mask[51:],encoder_outputs=(hidden_state[51:],None,None),decoder_attention_mask=decoder_attention_masks[51:100] , labels=candidate_ids[51:])['logits']),dim=0)
                    for i in range (1,int(num_negs/50)):
                        logits=torch.cat((logits,self.model(attention_mask=attention_mask[i*50+1:(i+1)*50+1],encoder_outputs=(hidden_state[i*50+1:(i+1)*50+1],None,None),decoder_attention_mask=decoder_attention_masks[i*50+1:(i+1)*50+1] , labels=candidate_ids[i*50+1:(i+1)*50+1])['logits']),dim=0)
                    # logits=candidate_output['logits']#(N+1)*T*H
                    # print(logits.shape)
                    scores=None
                    loss_fn=torch.nn.CrossEntropyLoss(ignore_index=-100).to(self.device)#((N+1),)
                    for logit,candidate_id in zip(logits,candidate_ids):
                        if scores is None:
                            scores=loss_fn(input=logit,target=candidate_id.long()).unsqueeze(0)
                            # print(scores)
                        else:
                            scores=torch.cat((scores,loss_fn(input=logit,target=candidate_id.long()).unsqueeze(0)),dim=0)#((N+1))
                    if scores_mark is None:
                        scores_mark=scores.unsqueeze(0)
                    else:
                        scores_mark=torch.cat((scores_mark,scores.unsqueeze(0)),dim=0)
                # print(scores_mark[:,:20])
                label_marks=torch.tensor([[1]+[0]*num_negs]).expand(len(inputs),-1)

                
                # print(scores_mark)
                # print(label_marks)
                # print('-------------')
                # print(scores_mark.size())
                # print(label_marks.size())
                # labels=np.array(labels)
                # labels=np.expand_dims(labels,axis=1)
                # candidates=np.append(labels,negs,axis=1)
                metrics,row,col= recalls_and_ndcgs_for_ks(scores_mark, label_marks, self.metric_ks)
                # best_three=candidates[row,col].reshape(-1,3)
                # print('')
                # print(labels)
                # print('----------')
                # print(best_three)
                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:3]]
                description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R')
                # description = description.format(*(average_meter_set[k].val for k in description_metrics))
                # # 上述用作测试，正式实验记得修改
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)         
            if self.args.local_rank==0:
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                self.logger_service.log_val(log_data)

            




    def test(self,accum_iter):
        self.model.eval()
        
        with torch.no_grad():
            average_meter_set=AverageMeterSet()
            tqdm_dataloader=tqdm(self.test_loader)

            for inputs,labels,negs in tqdm_dataloader:#negs:(N,B)
                num_negs=len(negs)
                negs=np.array(negs).T#B*N

                negs=negs.tolist()
                
                tokenized_inputs=self.tokenizer(inputs,max_length=512,padding='max_length',return_tensors="pt",truncation=True)
                input_ids=tokenized_inputs["input_ids"].to(self.device)#B*T
                attention_mask=tokenized_inputs["attention_mask"].to(self.device)#B*T

                tokenized_labels=self.tokenizer(labels,max_length=32,padding='max_length',return_tensors="pt",truncation=True)#B*T
                label_ids=tokenized_labels["input_ids"].to(self.device)
                decoder_attention_mask=tokenized_labels["attention_mask"].to(self.device)
                label_ids[label_ids == self.tokenizer.pad_token_id] = -100
                

                outputs=self.model(input_ids=input_ids,attention_mask=attention_mask,decoder_attention_mask=decoder_attention_mask , labels=label_ids,output_hidden_states=True,output_attentions=True)
                
                encoder_last_hidden_state=outputs['encoder_last_hidden_state']#B*T*H
                encoder_last_hidden_state=encoder_last_hidden_state.unsqueeze(1).expand(-1,num_negs+1,-1,-1)

                attention_masks=attention_mask.unsqueeze(1).expand(-1,num_negs+1,-1)

                neg_ids=None
                for neg in negs:#neg(N)
                    tokenized_neg=self.tokenizer(neg,max_length=32,padding='max_length',return_tensors="pt",truncation=True)#(N,T)
                    neg_id=tokenized_neg["input_ids"].to(self.device)
                    neg_decoder_attention_mask=tokenized_neg["attention_mask"].to(self.device)
                    neg_id[neg_id == self.tokenizer.pad_token_id] = -100
                    neg_id=neg_id.unsqueeze(0)
                    neg_decoder_attention_mask=neg_decoder_attention_mask.unsqueeze(0)
                    if neg_ids is None:
                        neg_ids=neg_id
                        neg_decoder_attention_masks=neg_decoder_attention_mask
                    else:
                        neg_ids=torch.cat((neg_ids,neg_id),dim=0)
                        neg_decoder_attention_masks=torch.cat((neg_decoder_attention_masks,neg_decoder_attention_mask),dim=0)

                label_ids=label_ids.unsqueeze(1)#(B,1,T)
                decoder_attention_mask=decoder_attention_mask.unsqueeze(1)    
                label_ids=torch.cat((label_ids,neg_ids),dim=1)#(B,(N+1),T)
                decoder_attention_mask=torch.cat((decoder_attention_mask,neg_decoder_attention_masks),dim=1)

                
                scores_mark=None
                for attention_mask,hidden_state,candidate_ids,decoder_attention_masks in zip(attention_masks,encoder_last_hidden_state,label_ids,decoder_attention_mask):#(N+1,T,H) (N+1,T)
                    # print(hidden_state.shape)
                    # print(candidate_ids.shape)#((N+1),T)
                    
                    logits=self.model(attention_mask=attention_mask[:51],encoder_outputs=(hidden_state[:51],None,None),decoder_attention_mask=decoder_attention_masks[:51] , labels=candidate_ids[:51])['logits']
                    logits=torch.cat((logits,self.model(attention_mask=attention_mask[51:],encoder_outputs=(hidden_state[51:],None,None),decoder_attention_mask=decoder_attention_masks[51:] , labels=candidate_ids[51:])['logits']),dim=0)
                    # logits=candidate_output['logits']#(N+1)*T*H
                    # print(logits.shape)
                    scores=None
                    loss_fn=torch.nn.CrossEntropyLoss(ignore_index=-100).to(self.device)#((N+1),)
                    for logit,candidate_id in zip(logits,candidate_ids):
                        if scores is None:
                            scores=loss_fn(input=logit,target=candidate_id.long()).unsqueeze(0)
                            # print(scores)
                        else:
                            scores=torch.cat((scores,loss_fn(input=logit,target=candidate_id.long()).unsqueeze(0)),dim=0)#((N+1))
                    if scores_mark is None:
                        scores_mark=scores.unsqueeze(0)
                    else:
                        scores_mark=torch.cat((scores_mark,scores.unsqueeze(0)),dim=0)
                
                label_marks=torch.tensor([[1]+[0]*num_negs]).expand(len(inputs),-1)


                labels=np.array(labels)
                labels=np.expand_dims(labels,axis=1)
                candidates=np.append(labels,negs,axis=1)
                metrics,row,col= recalls_and_ndcgs_for_ks(scores_mark, label_marks, self.metric_ks)
                best_three=candidates[row,col].reshape(-1,3)
                # print('')
                # print(labels)
                # print('----------')
                # print(best_three)
                
                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                # description_metrics = ['NDCG@%d' % k for k in self.metric_ks[3:]] +\
                #                       ['Recall@%d' % k for k in self.metric_ks[3:]]
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:3]]
                description = 'Test: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)
                
            
            if self.args.local_rank==0:
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                self.logger_service.log_test(log_data)

    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.train_batch_size and accum_iter != 0

    def _create_loggers(self):
        print(self.export_root)
        root = Path(self.export_root)
        writer = SummaryWriter(root.joinpath('logs_rank'+str(self.args.local_rank)))
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricGraphPrinter(writer, key='epoch', graph_name='Epoch', group_name='Train'),
            MetricGraphPrinter(writer, key='loss', graph_name='Loss', group_name='Train'),
        ]
        
        val_loggers = []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))
        val_loggers.append(RecentModelLogger(self.args,model_checkpoint))
        val_loggers.append(BestModelLogger(self.args,model_checkpoint, metric_key=self.best_metric))

        test_loggers = []
        for k in self.metric_ks:
            test_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='test'))
            test_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='test'))

        return writer, train_loggers,val_loggers
    
    
    def _create_state_dict(self):
        return {
            'model_state_dict': self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

    # def construct_trie(self,items):
    #     encoded_items=[]
    #     for item in items:
    #         encoded_items.append([0]+self.tokenizer.encode(item))
    #     prefix_trie=trie.MarisaTrie(encoded_items)
    #     return prefix_trie


class AbstractBaseLogger(metaclass=ABCMeta):
    @abstractmethod
    def log(self, *args, **kwargs):
        raise NotImplementedError

    def complete(self, *args, **kwargs):
        pass

class MetricGraphPrinter(AbstractBaseLogger):
    def __init__(self, writer, key='train_loss', graph_name='Train Loss', group_name='metric'):
        self.key = key
        self.graph_label = graph_name
        self.group_name = group_name
        self.writer = writer

    def log(self, *args, **kwargs):
        if self.key in kwargs:
            self.writer.add_scalar(self.group_name + '/' + self.graph_label, kwargs[self.key], kwargs['accum_iter'])
        else:
            self.writer.add_scalar(self.group_name + '/' + self.graph_label, 0, kwargs['accum_iter'])

    def complete(self, *args, **kwargs):
        self.writer.close()

class LoggerService(object):
    def __init__(self, train_loggers=None, val_loggers=None,test_loggers=None):
        self.train_loggers = train_loggers if train_loggers else []
        self.val_loggers = val_loggers if val_loggers else []
        self.test_loggers= test_loggers if test_loggers else []

    def complete(self, log_data):
        for logger in self.train_loggers:
            logger.complete(**log_data)
        for logger in self.val_loggers:
            logger.complete(**log_data)
        for logger in self.test_loggers:
            logger.complete(**log_data)    

    def log_train(self, log_data):
        for logger in self.train_loggers:
            logger.log(**log_data)

    def log_val(self, log_data):
        for logger in self.val_loggers:
            logger.log(**log_data)

    def log_test(self, log_data):
        for logger in self.test_loggers:
            logger.log(**log_data)

class RecentModelLogger(AbstractBaseLogger):
    def __init__(self,args, checkpoint_path, filename='checkpoint-recent.pth'):
        self.checkpoint_path = checkpoint_path
        # if not os.path.exists(self.checkpoint_path):
        if args.local_rank==0 and not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        self.recent_epoch = None
        self.filename = filename

    def log(self, *args, **kwargs):
        epoch = kwargs['epoch']

        if self.recent_epoch != epoch:
            self.recent_epoch = epoch
            state_dict = kwargs['state_dict']
            state_dict['epoch'] = kwargs['epoch']
            save_state_dict(state_dict, self.checkpoint_path, self.filename)

    def complete(self, *args, **kwargs):
        save_state_dict(kwargs['state_dict'], self.checkpoint_path, self.filename + '.final')


class BestModelLogger(AbstractBaseLogger):
    def __init__(self,args, checkpoint_path, metric_key='mean_iou', filename='best_acc_model.pth'):
        self.checkpoint_path = checkpoint_path
        # if not os.path.exists(self.checkpoint_path):
        if args.local_rank==0 and not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        self.best_metric = 0.
        self.metric_key = metric_key
        self.filename = filename

    def log(self, *args, **kwargs):
        current_metric = kwargs[self.metric_key]
        if self.best_metric < current_metric:
            print("Update Best {} Model at {}".format(self.metric_key, kwargs['epoch']))
            self.best_metric = current_metric
            save_state_dict(kwargs['state_dict'], self.checkpoint_path, self.filename)



def save_state_dict(state_dict, path, filename):
    torch.save(state_dict, os.path.join(path, filename))