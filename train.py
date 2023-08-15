# Copyright 2018 Dong-Hyun Lee, Kakao Brain.

""" Training Config & Helper Classes  """

import os
import json
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn

import checkpoint

from loguru import logger

from utils import plot_loss, Curve
from evaluate import Result

from typing import NamedTuple, Callable

class Config(NamedTuple):
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

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))


class Trainer(object):
    """Training Helper Class"""
    def __init__(self, cfg, model, data_iter, optimizer, save_dir, device):
        self.cfg = cfg # config for training : see class Config
        self.model = model
        self.data_iter = data_iter # iterator to load data
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.device = device # device name

    def train(self, get_loss, model_file=None, pretrain_file=None, data_parallel=True, fig_path: str = None, evaluate: Callable[[nn.Module, list], Result] = None):
        """ Train Loop """
        self.model.train() # train mode
        self.load(model_file, pretrain_file)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        loss_curve = Curve("loss", [])
        epoch_loss_curve = Curve("epoch_loss", [], []) # average loss in each epoch
        acc_curve = Curve("acc", [], [])
        
        logger.info('Start training!')
        global_step = 0 # global iteration steps regardless of epochs
        for e in range(self.cfg.n_epochs):
            model.train()
            loss_sum = 0. # the sum of iteration losses to get average loss in every epoch
            iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')
            for i, batch in enumerate(iter_bar):
                batch = [t.to(self.device) for t in batch]

                self.optimizer.zero_grad()
                loss = get_loss(model, batch, global_step).mean() # mean() for Data Parallelism
                loss.backward()
                self.optimizer.step()

                global_step += 1
                loss_sum += loss.item()
                iter_bar.set_description('Iter (loss=%5.3f)'%loss.item())
                
                loss_curve.y.append(loss.item())
                if epoch_loss_curve.x == []:
                    epoch_loss_curve.x.append(0)
                    epoch_loss_curve.y.append(loss.item())
                plot_loss([loss_curve, epoch_loss_curve], fig_path)

                if global_step % self.cfg.save_steps == 0: # save
                    self.save(global_step)

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    logger.success('Epoch %d/%d : Average Loss %5.3f' % (e+1, self.cfg.n_epochs, loss_sum/(i+1)))
                    logger.success('The Total Steps have been reached.')
                    self.save(global_step) # save and finish when global_steps reach total_steps
                    return

            if evaluate is not None:
                model.eval()
                logger.info('Start evaluation!')
                with torch.no_grad():
                    result = Result.reduce(map(lambda batch: evaluate(model, batch), tqdm(self.data_iter)))
                    
                total_acc, _ = result.summary()
                total_acc2, _ = result.summary(1)
                
                acc_curve.x.append(global_step)
                acc_curve.y.append(total_acc)
                logger.success(f'Accuracy: {total_acc:.4%} {total_acc2:.4%}')
                
                table = result.table()
                Result.heatmap(table, f'logs/heatmap/{datetime.now():%Y-%m-%d-%H-%M-%S}.jpg')
            
            epoch_loss_curve.x.append(global_step)
            epoch_loss_curve.y.append(loss_sum/(i+1))
            plot_loss([loss_curve, epoch_loss_curve], fig_path)
            logger.info('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
        self.save(global_step)

    def eval(self, evaluate: Callable[[nn.Module, list], Result], model_file, data_parallel=True):
        """ Evaluation Loop """
        self.model.eval() # evaluation mode
        self.load(model_file, None)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        logger.info('Start evaluation!')
        results = [] # prediction results
        iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')
        for batch in iter_bar:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad(): # evaluation without gradient calculation
                result = evaluate(model, batch) # accuracy to print
            results.append(result)
            accuracy, _ = result.summary(1)
            iter_bar.set_description('Iter(acc=%5.3f)'%accuracy)
        return Result.reduce(results)

    def load(self, model_file, pretrain_file):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            logger.info(f'Loading the model from {model_file}')
            self.model.load_state_dict(torch.load(model_file))

        elif pretrain_file: # use pretrained transformer
            logger.info(f'Loading the pretrained model from {pretrain_file}')
            if pretrain_file.endswith('.ckpt'): # checkpoint file in tensorflow
                checkpoint.load_model(self.model.transformer, pretrain_file)
            elif pretrain_file.endswith('.pt'): # pretrain model file in pytorch
                self.model.transformer.load_state_dict(
                    {key[12:]: value
                        for key, value in torch.load(pretrain_file).items()
                        if key.startswith('transformer')}
                ) # load only transformer parts


    def save(self, i):
        """ save current model """
        torch.save(self.model.state_dict(), # save model object before nn.DataParallel
            os.path.join(self.save_dir, 'model_steps_'+str(i)+'.pt'))

