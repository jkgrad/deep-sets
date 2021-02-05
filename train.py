import os
import time
import pickle
import random
import logging

import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from dataclasses import dataclass
from typing import Callable, List, Dict, Tuple, Union

from torch.utils.data import DataLoader

from module import DigitFuncApproximator, DigitFuncDataset

@dataclass
class Configuration:
    # Data Args
    n_train_samples: int = 60_000
    n_valid_samples: int = 10_000
    max_n_elements: int = 10
    function: Callable = torch.max

    # Dataset Path (if cached)
    train_path: str = 'data/train.pkl'
    valid_path: str = 'data/valid.pkl'
    log_path: str = 'checkpoint/train.log'
    save_path: str = 'checkpoint/best_model.pt'

    # Train Args
    device: Union[torch.device, str] = 'cpu'
    n_epochs: int = 10
    batch_sz: int = 128
    save_step: int = 100
    log_every: int = 100

    learning_rate: float = 1e-3
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

def train(args: "Configuration", logger: "logging.Logger"):
    device = args.device

    # Dataset
    if os.path.isfile(args.train_path) and os.path.isfile(args.valid_path):
        train_set = pickle.load(open(args,train_path, 'rb'))
        valid_set = pickle.load(open(args.valid_path, 'rb'))
    else:
        train_set = DigitFuncDataset(args.n_train_samples, args.max_n_elements, args.function)
        valid_set = DigitFuncDataset(args.n_valid_samples, args.max_n_elements, args.function)
        pickle.dump(train_set, open(args.train_path, 'wb'))
        pickle.dump(valid_set, open(args.valid_path, 'wb'))
    train_loader = tqdm(DataLoader(train_set, batch_size=args.batch_sz, shuffle=True))
    valid_loader = tqdm(DataLoader(valid_set, batch_size=args.batch_sz, shuffle=True))
    
    # Model
    model = DigitFuncApproximator(args.d_embed, args.d_hidden, args.activation)

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
    for epoch in range(args.n_epochs):
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
                    logger.info('Best validation metric found ... Saving ...')
                    torch.save(model.state_dict(), args.train_path)

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
            crt += torch.sum(preds == labels).item()
            tot += len(labels)
    acc = crt / tot
    return acc
            
if __name__ == '__main__':
    args = Configuration()
    logging.basicConfig(filename=args.log_path,
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger()

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