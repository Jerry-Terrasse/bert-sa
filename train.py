# Copyright 2018 Dong-Hyun Lee, Kakao Brain.

""" Training Config & Helper Classes  """

import os
import json
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

import checkpoint

from loguru import logger

from utils import plot_loss, Curve
from evaluate import Result

from typing import NamedTuple, Callable

class Config(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    batch_size: int = 32
    lr: float = 5e-5 # learning rate
    n_epochs: int = 10 # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup: float = 0.1
    save_steps: int = 100 # interval for saving model
    total_steps: int = 100000 # total number of steps to train

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))

class TomatoConfig(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    batch_size: int = 32
    eval_batch_size: int = 32
    lr: float = 5e-5 # learning rate
    n_epochs: int = 10 # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup: float = 0.1
    save_steps: int = 100 # interval for saving model
    save_per_epoch: bool = True # save model per epoch
    total_steps: int = 100000 # total number of steps to train
    train_data: str = 'home_train.json'
    eval_data: str = 'home_eval.json'

class Trainer(object):
    """Training Helper Class"""
    def __init__(self, cfg: TomatoConfig, model, data_iter, optimizer, save_dir, device):
        self.cfg = cfg # config for training : see class Config
        self.model = model
        self.data_iter = data_iter # iterator to load data
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.device = device # device name

    def train(
        self,
        get_loss: Callable[[nn.Module, list, int], torch.Tensor],
        log_dir: str,
        model_file: str = None,
        pretrain_file: str = None,
        data_parallel: bool = True,
        fig_path: str = None,
        evaluate: Callable[[nn.Module, list], Result] = None,
        eval_iter: DataLoader = None
    ):
        """ Train Loop """
        writer = SummaryWriter(log_dir=log_dir) # writer for tensorboard
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
                if fig_path:
                    plot_loss([loss_curve, epoch_loss_curve], fig_path)
                writer.add_scalar('loss', loss.item(), global_step)

                if global_step % self.cfg.save_steps == 0: # save
                    self.save(f"model_step_{global_step}.pt")

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    logger.success('Epoch %d/%d : Average Loss %5.3f' % (e+1, self.cfg.n_epochs, loss_sum/(i+1)))
                    logger.success('The Total Steps have been reached.')
                    self.save(f"model_step_{global_step}.pt") # save and finish when global_steps reach total_steps
                    return

            if self.cfg.save_per_epoch:
                self.save(f"model_epoch{e+1}.pt")
            
            if evaluate is not None:
                model.eval()
                logger.info('Start evaluation!')
                with torch.no_grad():
                    result = Result.reduce(map(lambda batch: evaluate(model, batch), tqdm(eval_iter)))
                
                eval_loss = result.loss.mean() if result.loss is not None else None
                if eval_loss:
                    writer.add_scalar('eval_loss', eval_loss.item(), global_step)
                    logger.info(f'Evaluation Loss: {eval_loss:.4f}')
                
                total_acc, _ = result.summary()
                total_acc2, _ = result.summary(1)
                
                acc_curve.x.append(global_step)
                acc_curve.y.append(total_acc.item())
                writer.add_scalar('acc', total_acc.item(), global_step)
                writer.add_scalar('acc2', total_acc2.item(), global_step)
                logger.success(f'Accuracy: {total_acc:.4%} {total_acc2:.4%}')
                
                table = result.table()
                img = Result.heatmap(table, f'logs/heatmap/{datetime.now():%Y-%m-%d-%H-%M-%S}.jpg')
                writer.add_image('heatmap', img, global_step, dataformats='HWC')
                table_ratio = table / table.sum(dim=1, keepdim=True)
                img = Result.heatmap(table_ratio, f'logs/heatmap/ratio-{datetime.now():%Y-%m-%d-%H-%M-%S}.jpg')
                writer.add_image('heatmap_ratio', img, global_step, dataformats='HWC')
            
            epoch_loss_curve.x.append(global_step)
            epoch_loss_curve.y.append(loss_sum / self.cfg.n_epochs)
            writer.add_scalar('epoch_loss', loss_sum / self.cfg.n_epochs, global_step)
            if fig_path:
                plot_loss([loss_curve, epoch_loss_curve], fig_path)
            logger.info('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum / self.cfg.n_epochs))
        if not self.cfg.save_per_epoch:
            self.save(f"model_step_{global_step}.pt")

    def eval(self, evaluate: Callable[[nn.Module, list], Result], log_dir: str, model_file, data_parallel=True):
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


    def save(self, fname: str):
        """ save current model """
        save_path = os.path.join(self.save_dir, fname)
        logger.info(f'Saving the model to {save_path}')
        torch.save(
            self.model.state_dict(), # save model object before nn.DataParallel
            save_path
        )

