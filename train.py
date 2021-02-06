import os
import time
import pickle
import random
import logging
import inspect

import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from dataclasses import dataclass
from typing import Callable, List, Dict, Tuple, Union

from torch.utils.data import DataLoader

from data import DigitFuncDataset
from module import DigitFuncApproximator

@dataclass
class Configuration:
    # Data Args
    n_train_samples: int = 100_000
    n_valid_samples: int = 10_000
    max_n_elements: int = 10
    function: Callable = lambda x: torch.sum(x[x > 0], dim=-1)

    # Dataset Path (if cached)
    train_path: str = 'data/train.pkl'
    valid_path: str = 'data/valid.pkl'
    save_path: str = 'checkpoint'

    # Train Args
    device: Union[torch.device, str] = 'cuda'
    n_epochs: int = 10
    batch_sz: int = 128
    save_step: int = 100
    log_every: int = 100

    learning_rate: float = 1e-4
    gradient_accumulation_step: int = 1
    weight_decay: float = 0.0

    optimizer: str = 'Adam'
    criterion: str = 'L1Loss'

    # Model Args
    d_embed: int = 100
    d_hidden: int = 30
    activation: str = 'Tanh'

    # Seed
    seed: int = 42

    def __str__(self):
        config_str = ''
        for k,v in self.__dict__.items():
            if k == 'function':
                func_str = inspect.getsource(v)
                v = func_str.replace('function: Callable = ', '').strip()
            config_str += '{} = {}\n'.format(k,v)
        return config_str


def train(args: "Configuration", logger: "logging.Logger"):
    device = args.device

    # Dataset
    train_set = DigitFuncDataset(args.n_train_samples, args.max_n_elements, args.function, args.train_path)
    valid_set = DigitFuncDataset(args.n_valid_samples, args.max_n_elements, args.function, args.valid_path)

    train_loader = DataLoader(train_set, batch_size=args.batch_sz, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_sz, shuffle=False)
    
    # Model
    model = DigitFuncApproximator(args.d_embed, args.d_hidden, args.activation)
    logger.info(str(model))

    # Loss
    criterion  = getattr(nn, args.criterion)()
    
    # Optimizer
    optimizer = getattr(torch.optim, args.optimizer)
    if args.optimizer != 'AdamW':
        optimizer = optimizer(model.parameters(), lr=args.learning_rate)
    else:
        no_decay  = ['bias', 'LayerNorm.weight']
        grouped_params = [
            {
                'params': [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay
            },
            {
                'params': [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = optimizer(grouped_params, lr=args.learning_rate)

    model.to(device)
    best_metric = 0.
    for epoch in tqdm(range(args.n_epochs)):
        logger.info('Begin Epoch {} ...'.format(epoch))
        for idx, batch in enumerate(train_loader):
            inputs = batch['input'].to(device)
            labels = batch['label']

            logits = model(inputs).cpu()

            loss = criterion(logits.squeeze(-1), labels)
            loss = loss / args.gradient_accumulation_step
            loss.backward()

            if not (idx+1) % args.log_every:
                logger.info('Epoch {} Loss : {}'.format(epoch + (idx+1) / len(train_loader), loss.item()))

            if not (idx+1) % args.gradient_accumulation_step:
                optimizer.step()
                optimizer.zero_grad()

            if not (idx+1) % args.save_step:
                metric = validate(args, model, valid_loader)
                logger.info('Validation Accuracy : {}'.format(metric))

                if best_metric < metric:
                    best_metric = metric
                    logger.info('Best validation metric found ... Saving ...')
                    torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model.pt'))

def validate(args: "Configuration", model: "nn.Module", valid_loader: "DataLoader"):
    device = args.device
    model.to(device)
    model.eval()
    crt, tot = 0, 0
    with torch.no_grad():
        for idx, batch in enumerate(valid_loader):
            inputs = batch['input'].to(device)
            labels = batch['label']
            # Round logits to nearest integer (MAE Loss Training)
            logits = model(inputs).cpu()
            preds  = torch.round(logits)
            # Correct and total examples
            crt += torch.sum(preds.squeeze() == labels.squeeze()).item()
            tot += len(labels)
    acc = crt / tot
    return acc

def get_logger(args: "Configuration"):
    current_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.gmtime())   
    logging.basicConfig(
        filename='{}/{}_train.log'.format(args.save_path, current_time), 
        filemode='w', 
        format="%(asctime)s | %(filename)15s | %(levelname)7s | %(funcName)10s | %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    return logger

if __name__ == '__main__':
    args = Configuration()
    logger = get_logger(args)

    logger.info('Configurations ...')
    logger.info('{}'.format(str(args)))

    if args.seed < 0:
        logger.info('No seed will be set.')
    else:
        logger.info('Setting seed to {} ...'.format(args.seed))
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    start_time = time.time()
    logger.info('Begin Training ...')
    train(args, logger)
    logger.info('Training Finished ... Time Elapsed : {} min(s)'.format((time.time() - start_time) / 60))