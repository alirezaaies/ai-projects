from custom_dataset import TextClassificationDataset, collate_fn, load_data, build_vocab
from torch.utils.data import DataLoader
from tokenizers import my_tokenizer_number
import  torch
from trainer_modules import train_epoch, evaluate
from models import (
    BasicRNN,
    RNNClassifier
)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import re
import string
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

BATCH_SIZE = 64
MAX_LEN = 200

dataset_local = load_data(source='LOCAL')

# Assume dataset_local['train'] exists
vocab = build_vocab(dataset_local['train'], text_column='text', vocab_size=25000, min_freq=2)

train_text_data = dataset_local['train']['text']
valid_text_data = dataset_local['validation']['text']
test_text_data = dataset_local['test']['text']

train_text_indices_data = list(my_tokenizer_number(train_text_data, vocab))
valid_text_indices_data = list(my_tokenizer_number(valid_text_data, vocab))
test_text_indices_data = my_tokenizer_number(test_text_data, vocab)

train_label_data = list(dataset_local['train']['label'])    
valid_label_data = dataset_local['validation']['label']
test_label_data = dataset_local['test']['label']

train_dataset = TextClassificationDataset(train_text_indices_data, train_label_data, vocab, max_len=MAX_LEN)
valid_dataset  = TextClassificationDataset(valid_text_indices_data, valid_label_data, vocab, max_len=MAX_LEN)
test_dataset  = TextClassificationDataset(test_text_indices_data, test_label_data, vocab, max_len=MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader  = DataLoader(valid_dataset, batch_size=BATCH_SIZE*2)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE*2)
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}")
print(f"Number of testing samples: {len(test_dataset)}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# print(next(iter(train_loader)))  # now works

# data = next(iter(train_loader))


EMBED_DIM    = 128          # increased a bit
HIDDEN_DIM   = 256
NUM_LAYERS   = 1            # start with 1
DROPOUT      = 0.4          # slightly higher
BIDIRECTIONAL = True       # try True later (stronger but 2× slower)
LEARNING_RATE = 0.0008      # lower than before
N_EPOCHS     = 8            # LSTMs usually converge faster

# Re-create model
model_rnn = RNNClassifier(
    vocab_size=len(vocab),
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=1,  # binary classification
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    bidirectional=BIDIRECTIONAL
).to(device)



optimizer = optim.Adam(model_rnn.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)  # small L2
criterion = nn.BCEWithLogitsLoss()


for epoch in range(1, N_EPOCHS + 1):
    train_loss, train_acc = train_epoch(model_rnn, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model_rnn, test_loader, criterion, device)
    
    print(f"Epoch {epoch}/{N_EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.4f}")
    print("-" * 60)