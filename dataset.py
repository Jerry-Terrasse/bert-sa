import json
from collections import Counter
from multiprocessing import Pool
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import tokenization

from loguru import logger

def init(vocab_file: str, do_lower_case: bool = True, max_len_ = 100):
    global tokenizer, max_len
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
    max_len = max_len_

def quote2tokens(quote: str):
    tokens = ['[CLS]'] + tokenizer.tokenize(quote, max_len)
    if len(tokens) >= max_len:
        tokens = tokens[:max_len]
    else:
        tokens += ['[PAD]'] * (max_len - len(tokens))
    # print(tokens)
    # breakpoint()
    tokens = tokenizer.convert_tokens_to_ids(tokens)
    # tokens = torch.tensor(tokens)
    return tokens

class TomatoDataset(Dataset):
    """
    Review Dataset of RottenTomatoes
    
    Tuple[float, str]
    """
    def __init__(self, path: str, vocab_file: str, do_lower_case: bool = True, max_len: int = 100, size: int = -1):
        super().__init__()
        max_len -= 1 # for [CLS]
        
        self.data = json.load(open(path, 'r'))
        if size > 0:
            self.data = self.data[:size]
        # self.data = json.load(open(path, 'r'))[:1000] # for test
        self.ratings: list[float] = [r for r, q in self.data]
        self.quotes: list[str] = [q for r, q in self.data]
        self.counter = Counter(self.ratings)
        logger.info(self.counter)
        
        logger.info('Loading dataset...')
        self.rating_tensors = [torch.tensor(r / 5.5) for r in self.ratings]
        tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
        self.tokenizer = tokenizer

        # to view dataset
        # init(vocab_file, do_lower_case, max_len)
        # token_list = list(map(quote2tokens, self.quotes))
        
        chunksize = len(self.quotes) // 10 // 8
        with Pool(10, initializer=init, initargs=(vocab_file, do_lower_case, max_len)) as p:
            token_list = list(tqdm(p.imap(quote2tokens, self.quotes, chunksize=chunksize), total=len(self.quotes)))
        
        self.quote_tensor = torch.tensor(token_list)
        self.mask_tensor = (self.quote_tensor != tokenizer.vocab['[PAD]']).long()
        logger.success('Dataset loaded.')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.rating_tensors[index], self.quote_tensor[index], self.mask_tensor[index]