# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Fine-tuning on A Classification Task with pretrained Transformer """

import itertools
import csv
import fire
import json
from collections import Counter
from typing import NamedTuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import tokenization
import models
import optim
import train

from utils import set_seeds, get_device, truncate_tokens_pair

from tqdm import tqdm
# from loguru import logger

class TomatoDataset(Dataset):
    """
    Review Dataset of RottenTomatoes
    
    Tuple[float, str]
    """
    def __init__(self, path: str, vocab_file: str, do_lower_case: bool = True, max_len: int = 100):
        super().__init__()
        max_len -= 1 # for [CLS]
        
        self.data = json.load(open(path, 'r'))
        self.ratings: list[float] = [r for r, q in self.data]
        self.quotes: list[str] = [q for r, q in self.data]
        self.counter = Counter(self.ratings)
        print(self.counter)
        
        print('Loading dataset...')
        self.rating_tensors = [torch.tensor(r / 5.) for r in self.ratings]
        self.quote_tensors = []
        self.mask_tensors = []
        tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
        self.tokenizer = tokenizer
        for quote in tqdm(self.quotes):
            tokens = ['[CLS]'] + tokenizer.tokenize(quote, max_len)
            if len(tokens) >= max_len:
                tokens = tokens[:max_len]
            else:
                tokens += ['[PAD]'] * (max_len - len(tokens))
            tokens = tokenizer.convert_tokens_to_ids(tokens)
            tokens = torch.tensor(tokens)
            self.quote_tensors.append(tokens)
            
            mask = (tokens != self.tokenizer.vocab['[PAD]']).long()
            self.mask_tensors.append(mask)
        print('Done')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.rating_tensors[index], self.quote_tensors[index], self.mask_tensors[index]

class Predictor(nn.Module):
    """ Predict rating from quote """
    def __init__(self, cfg: models.Config):
        super().__init__()
        self.transformer = models.Transformer(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.predictor = nn.Linear(cfg.dim, 1)
        self.output = nn.Sigmoid()
    
    def forward(self, input_ids, input_mask):
        segment_ids = torch.zeros_like(input_ids, dtype=torch.long)
        h = self.transformer(input_ids, segment_ids, input_mask)
        pooled_h = self.activ(self.fc(h[:, 0]))
        output = self.output(self.predictor(self.drop(pooled_h)))
        return output

class TomatoConfig(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    batch_size: int = 32
    lr: int = 5e-5 # learning rate
    n_epochs: int = 10 # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup: float = 0.1
    save_steps: int = 100 # interval for saving model
    total_steps: int = 100000 # total number of steps to train
    train_data: str = 'home_train.json'
    eval_data: str = 'home_eval.json'

def discriminate(X: torch.Tensor):
    return torch.round(X * 10).long()

def main(
         train_cfg='config/train_tomato.json',
         model_cfg='config/bert_base.json',
         model_file=None,
         pretrain_file='../data/BERT_pretrained/uncased_L-12_H-768_A-12/bert_model.ckpt',
         data_parallel=True,
         vocab='../data/BERT_pretrained/uncased_L-12_H-768_A-12/vocab.txt',
         save_dir='save/',
         max_len=100,
         mode='train',
         total_steps=-1):

    train_cfg_dict = json.load(open(train_cfg))
    if total_steps > 0:
        train_cfg_dict['total_steps'] = total_steps
    cfg = TomatoConfig(**train_cfg_dict)
    
    model_cfg_dict = json.load(open(model_cfg))
    model_cfg = models.Config(**model_cfg_dict)
    
    set_seeds(cfg.seed)
    
    train_file = cfg.train_data
    eval_file = cfg.eval_data
    train_data, eval_data = TomatoDataset(train_file, vocab, max_len=max_len), TomatoDataset(eval_file, vocab, max_len=max_len)
    train_iter, eval_iter = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True), DataLoader(eval_data, batch_size=cfg.batch_size, shuffle=False)
    
    model = Predictor(model_cfg)
    criterion = nn.MSELoss()

    trainer = train.Trainer(cfg,
                            model,
                            train_iter if mode == 'train' else eval_iter,
                            optim.optim4GPU(cfg, model),
                            save_dir, get_device())

    if mode == 'train':
        def get_loss(model, batch, global_step): # make sure loss is a scalar tensor
            rating, quote, mask = batch
            prediction = model(quote, mask).reshape(-1)
            loss = criterion(prediction, rating)
            return loss

        trainer.train(get_loss, model_file, pretrain_file, data_parallel)

    elif mode == 'eval':
        def evaluate(model, batch):
            rating, quote, mask = batch
            prediction = model(quote, mask)
            
            pred_level = discriminate(prediction)
            label_level = discriminate(rating)
            
            result = (pred_level == label_level).float() #.cpu().numpy()
            accuracy = result.mean()
            return accuracy, result

        results = trainer.eval(evaluate, model_file, data_parallel)
        total_accuracy = torch.cat(results).mean().item()
        print('Accuracy: ', total_accuracy)


if __name__ == '__main__':
    fire.Fire(main)
