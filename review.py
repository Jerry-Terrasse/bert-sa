# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Fine-tuning on A Classification Task with pretrained Transformer """

import sys
import fire
import json
from collections import Counter
from typing import NamedTuple
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

import models
import optim
import train

from utils import set_seeds, get_device, truncate_tokens_pair
from evaluate import Result

from tqdm import tqdm
from loguru import logger
logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
title = sys.argv[0].split("/")[-1].split(".")[0]
prefix = f"logs/{title}-{datetime.now():%Y-%m-%d-%H-%M-%S}"
log_file = f"{prefix}.log"
logger.add(log_file, colorize=False)

from dataset import TomatoDataset

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
    eval_batch_size: int = 32
    lr: int = 5e-5 # learning rate
    n_epochs: int = 10 # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup: float = 0.1
    save_steps: int = 100 # interval for saving model
    total_steps: int = 100000 # total number of steps to train
    train_data: str = 'home_train.json'
    eval_data: str = 'home_eval.json'

def main(
         train_cfg='config/train_tomato.json',
         model_cfg='config/bert_base.json',
         model_file=None,
         eval_model='save/model_steps_18000.pt',
         pretrain_file='../data/BERT_pretrained/uncased_L-12_H-768_A-12/bert_model.ckpt',
         data_parallel=True,
         vocab='../data/BERT_pretrained/uncased_L-12_H-768_A-12/vocab.txt',
         save_dir='save/exp1.1',
         max_len=100,
         mode='train',
         total_steps=-1):
    logger.info(f"{mode} Mode")

    train_cfg_dict = json.load(open(train_cfg))
    if total_steps > 0:
        train_cfg_dict['total_steps'] = total_steps
    cfg = TomatoConfig(**train_cfg_dict)
    
    model_cfg_dict = json.load(open(model_cfg))
    model_cfg = models.Config(**model_cfg_dict)
    
    logger.info(f"Train Config: {cfg}")
    logger.info(f"Model Config: {model_cfg}")
    logger.info(f"argv: {sys.argv}")
    logger.debug(f"main args: {train_cfg=} {model_cfg=} {model_file=} {eval_model=} {pretrain_file=} {data_parallel=} {vocab=} {save_dir=} {max_len=} {mode=} {total_steps=}")
    
    set_seeds(cfg.seed)
    
    train_file = cfg.train_data
    eval_file = cfg.eval_data
    train_data, eval_data = TomatoDataset(train_file, vocab, max_len=max_len), TomatoDataset(eval_file, vocab, max_len=max_len)
    
    # Class Balance
    weights = {rating: len(train_data)/cnt for rating, cnt in train_data.counter.items()}
    weights = [weights[rating] for rating in train_data.ratings]
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, len(weights))
    
    train_iter, eval_iter = DataLoader(train_data, batch_size=cfg.batch_size, sampler=sampler), DataLoader(eval_data, batch_size=cfg.eval_batch_size, shuffle=False)
    
    # test balance
    # tot = []
    # for batch in train_iter:
    #     rating, quote, mask = batch
    #     tot.extend(rating.tolist())
    # print(f"Sample: {Counter(tot)}")    
    # exit(0)
    
    model = Predictor(model_cfg)
    criterion = nn.MSELoss()
    logger.info(f"Model: \n{model}")
    logger.info(f"Criterion: {criterion}")
    logger.info(f"Number of parameters = {sum(p.numel() for p in model.parameters())}, size = {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024} MB")

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

        trainer.train(get_loss, model_file, pretrain_file, data_parallel, f'{prefix}.jpg')

    elif mode == 'eval':
        def evaluate(model, batch):
            rating, quote, mask = batch
            prediction = model(quote, mask).reshape(-1)
            
            result = Result(rating, prediction)
            
            return result

        result = trainer.eval(evaluate, eval_model, data_parallel)
        total_acc, _ = result.summary()
        total_acc2, _ = result.summary(1)
        logger.success(f'Accuracy: {total_acc:.4%} {total_acc2:.4%}')
        table = result.table()
        Result.heatmap(table, f'{prefix}_heatmap.jpg')
    
    else:
        logger.error(f'invalid mode: {mode}')


if __name__ == '__main__':
    fire.Fire(main)
