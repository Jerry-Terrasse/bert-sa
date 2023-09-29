# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Fine-tuning on A Classification Task with pretrained Transformer """

import sys
import fire
import json
from datetime import datetime
import os

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
if __name__ == '__main__':
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    title = sys.argv[0].split("/")[-1].split(".")[0]
    prefix = f"logs/{title}-{datetime.now():%Y-%m-%d-%H-%M-%S}"
    log_file = f"{prefix}.log"
    logger.add(log_file, colorize=False)

from dataset import TomatoDataset
from tokenization import load_vocab

class Predictor(nn.Module):
    """ Predict rating from quote """
    def __init__(self, cfg: models.Config):
        super().__init__()
        self.transformer = models.Transformer(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.predictor = nn.Linear(cfg.dim, 1)
        self.output = nn.ReLU()
    
    def forward(self, input_ids, input_mask):
        segment_ids = torch.zeros_like(input_ids, dtype=torch.long)
        h = self.transformer(input_ids, segment_ids, input_mask)
        pooled_h = self.activ(self.fc(h[:, 0]))
        output = self.output(self.predictor(self.drop(pooled_h)))
        return output

def infer(model: nn.Module, model_file: str, data_iter: DataLoader, vocab_file: str, data_parallel: bool, device):
    model.eval()
    model.load_state_dict(torch.load(model_file))
    model.to(device)
    if data_parallel:
        model = nn.DataParallel(model)
    
    vocab = load_vocab(vocab_file)
    id2word = {v: k for k, v in vocab.items()}
    def ids2text(ids: torch.Tensor):
        return ' '.join([id2word[id] for id in ids.tolist()])
    
    logger.info("Start Inference!")
    preds = []
    for batch in tqdm(data_iter):
        _, quote, mask = [x.to(device) for x in batch]
        with torch.no_grad():
            prediction = model(quote, mask).reshape(-1)
        preds.append(prediction.tolist())
        # for p, q, m in zip(prediction, quote.to('cpu'), mask.to('cpu')):
        #     breakpoint()
        #     print(q[m])
    results = []
    for pred, batch in zip(preds, data_iter):
        _, quote, mask = batch
        result = [
            (p, ids2text(q[m.bool()]))
            for p, q, m in zip(pred, quote, mask)
        ]
        results.extend(result)
    return results

@logger.catch(reraise=True)
def main(
    train_cfg='config/train_tomato_g.json',
    model_cfg='config/bert_base.json',
    model_file=None,
    eval_model='save/model_steps_18000.pt',
    pretrain_file='../data/BERT_pretrained/uncased_L-12_H-768_A-12/bert_model.ckpt',
    data_parallel=True,
    vocab='../data/BERT_pretrained/uncased_L-12_H-768_A-12/vocab.txt',
    save_dir='save/exp1.11',
    log_dir='logs/tb/exp1.11',
    max_len=100,
    dataset_size=-1, # -1 for full dataset, otherwise for partial dataset for debugging
    mode='train',
    eval_in_train=True,
):
    logger.info(f"{mode} Mode")

    train_cfg_dict = json.load(open(train_cfg))
    cfg = train.TomatoConfig(**train_cfg_dict)
    
    model_cfg_dict = json.load(open(model_cfg))
    model_cfg = models.Config(**model_cfg_dict)
    
    logger.info(f"Train Config: {cfg}")
    logger.info(f"Model Config: {model_cfg}")
    logger.info(f"argv: {sys.argv}")
    logger.debug(f"main args: {train_cfg=} {model_cfg=} {model_file=} {eval_model=} {pretrain_file=} {data_parallel=} {vocab=} {save_dir=} {max_len=} {mode=}")
    
    set_seeds(cfg.seed)
    
    if not os.path.exists(save_dir):
        logger.warning(f"save_dir {save_dir} not exists, create it")
        os.makedirs(save_dir)
    if not os.path.exists(log_dir):
        logger.warning(f"log_dir {log_dir} not exists, create it")
        os.makedirs(log_dir)
    
    train_file = cfg.train_data
    eval_file = cfg.eval_data
    train_data, eval_data = TomatoDataset(train_file, vocab, max_len=max_len, size=dataset_size), TomatoDataset(eval_file, vocab, max_len=max_len, size=dataset_size)
    
    # Class Balance
    weights = {rating: len(train_data)/cnt for rating, cnt in train_data.counter.items()}
    weights = [weights[rating] for rating in train_data.ratings]
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

    device = get_device()
    total_steps = cfg.total_steps if cfg.total_steps > 0 else cfg.n_epochs * len(train_iter)
    train_cfg_dict['total_steps'] = total_steps
    cfg = train.TomatoConfig(**train_cfg_dict)
    trainer = train.Trainer(cfg,
                            model,
                            train_iter if mode == 'train' else eval_iter,
                            optim.optim4GPU(cfg, model),
                            save_dir, device)

    def get_loss(model, batch, global_step): # make sure loss is a scalar tensor
        rating, quote, mask = [x.to(device) for x in batch]
        prediction = model(quote, mask).reshape(-1)
        loss = criterion(prediction, rating)
        return loss
    def evaluate(model, batch):
        rating, quote, mask = [x.to(device) for x in batch]
        prediction = model(quote, mask).reshape(-1)
        loss = criterion(prediction, rating).reshape(-1)
        result = Result(rating, prediction, loss)
        return result
    
    if mode == 'train':
        trainer.train(get_loss, log_dir, model_file, pretrain_file, data_parallel, f'{prefix}.jpg', evaluate if eval_in_train else None, eval_iter)

    elif mode == 'eval':
        result = trainer.eval(evaluate, log_dir, eval_model, data_parallel)
        total_acc, _ = result.summary()
        total_acc2, _ = result.summary(1)
        logger.success(f'Accuracy: {total_acc:.4%} {total_acc2:.4%}')
        table = result.table()
        Result.heatmap(table, f'{prefix}_heatmap.jpg')
        
        table_ratio = table / table.sum(dim=1, keepdim=True)
        Result.heatmap(table_ratio, f'{prefix}_heatmap_ratio.jpg', 0., .5)
    
    elif mode == 'infer':
        result = infer(model, eval_model, eval_iter, vocab, data_parallel, device)
        json.dump(result, open(f'result.json', 'w'))
    
    else:
        logger.error(f'invalid mode: {mode}')


if __name__ == '__main__':
    fire.Fire(main)
