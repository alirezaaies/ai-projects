from custom_dataset import CustomDataset, collate_fn, load_data, build_vocab
from torch.utils.data import DataLoader
from test_model import predict_sentence
from tokenizers import my_tokenizer_number
import  torch
from trainer_modules import train_epoch, evaluate
from models import (
    # BasicRNN,
    RNNClassifier,
    LSTMClassifier,
    CNNClassifier,
    TransformerClassifier
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
import os
import pandas as pd

BATCH_SIZE = 64
MAX_LEN = 200

# load data and build vocab
dataset_local = load_data(source='LOCAL')
# Assume dataset_local['train'] exists
vocab = build_vocab(dataset_local['train'], text_column='text', vocab_size=25000, min_freq=2)

# get text data for each split
train_text_data = dataset_local['train']['text']
valid_text_data = dataset_local['validation']['text']
test_text_data = dataset_local['test']['text']

# tokenizing text data to indices using the number-based tokenizer and the built vocab
train_text_indices_data = list(my_tokenizer_number(train_text_data, vocab))
valid_text_indices_data = list(my_tokenizer_number(valid_text_data, vocab))
test_text_indices_data = list(my_tokenizer_number(test_text_data, vocab))

# get labels
train_label_data = list(dataset_local['train']['label'])    
valid_label_data = list(dataset_local['validation']['label'])
test_label_data = list(dataset_local['test']['label'])

# create datasets and dataloaders
train_dataset = CustomDataset(train_text_indices_data, train_label_data, vocab, max_len=MAX_LEN)
valid_dataset  = CustomDataset(valid_text_indices_data, valid_label_data, vocab, max_len=MAX_LEN)
test_dataset  = CustomDataset(test_text_indices_data, test_label_data, vocab, max_len=MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader  = DataLoader(valid_dataset, batch_size=BATCH_SIZE*2)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE*2)

# print dataset sizes to verify everything is loaded correctly
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}")
print(f"Number of testing samples: {len(test_dataset)}")

# check that you have GPU or not
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


# set constants for the model and training
EMBED_DIM     = 128          # increased a bit
HIDDEN_DIM    = 256
NUM_LAYERS    = 1            # start with 1
DROPOUT       = 0.4          # slightly higher
BIDIRECTIONAL = True       # try True later (stronger but 2× slower)
LEARNING_RATE = 0.0008      # lower than before
N_EPOCHS      = 50            # LSTMs usually converge faster

# Folder to save results
RESULTS_DIR = "results_transformer_classifier"
os.makedirs(RESULTS_DIR, exist_ok=True)

# define your model - you can change the model class and parameters based on models in models.py
# model = RNNClassifier(
#     vocab_size=len(vocab),
#     embed_dim=EMBED_DIM,
#     hidden_dim=HIDDEN_DIM,
#     num_classes=1,  # binary classification
#     num_layers=NUM_LAYERS,
#     dropout=DROPOUT,
#     bidirectional=BIDIRECTIONAL
# ).to(device) # send model to device

# swap models easily
model_name = "transformer"  # change to "rnn","lstm", "cnn", "transformer"
if model_name == "rnn":
    model = RNNClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=1,
        num_layers=NUM_LAYERS,
        dropout=0.4,
        bidirectional=BIDIRECTIONAL
    ).to(device)
elif model_name == "lstm":
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=1,
        num_layers=NUM_LAYERS,
        dropout=0.4,
        bidirectional=BIDIRECTIONAL
    ).to(device)
elif model_name == "cnn":
    model = CNNClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        num_classes=1,
        kernel_sizes=[3,4,5],
        num_filters=100,
        dropout=0.5
    ).to(device)
elif model_name == "transformer":
    model = TransformerClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        num_classes=1,
        num_heads=4,
        hidden_dim=HIDDEN_DIM,
        num_layers=2,
        dropout=0.2,
        max_len=500
    ).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)  # small L2
loss_fn = nn.BCEWithLogitsLoss()


# History dict
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

best_val_acc = 0.0
best_val_loss = float('inf')
# best_model_path = os.path.join(RESULTS_DIR, "best_model.pth")

for epoch in range(1, N_EPOCHS + 1):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
    val_loss, val_acc = evaluate(model, test_loader, loss_fn, device)
    
    # Save history
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    
    # Paths to save best models
    best_loss_path = os.path.join(RESULTS_DIR, "best_model_by_loss.pth")
    best_acc_path  = os.path.join(RESULTS_DIR, "best_model_by_acc.pth")

    # Save by loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_loss_path)
        print(f"Saved model by val_loss={val_loss:.4f} at epoch {epoch}")

    # Save by accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_acc_path)
        print(f"Saved model by val_acc={val_acc:.4f} at epoch {epoch}")
    
    
    print(f"Epoch {epoch}/{N_EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.4f}")
    print("-" * 60)
    

# Save history to disk
import json
history_path = os.path.join(RESULTS_DIR, "history.json")
with open(history_path, "w") as f:
    json.dump(history, f)
print(f"Saved training history to {history_path}")
    
    
# Plot loss
plt.figure(figsize=(8,6))
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.grid(True)
# Set y-axis limits
plt.ylim(0, 1)  # min=0, max=2
plt.savefig(os.path.join(RESULTS_DIR, "loss_plot.png"))
plt.close()

# Plot accuracy
plt.figure(figsize=(8,6))
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.grid(True)
# Set y-axis limits
plt.ylim(0.4, 1)  # min=0, max=2
plt.savefig(os.path.join(RESULTS_DIR, "acc_plot.png"))
plt.close()

print(f"Saved plots to {RESULTS_DIR}")



# Load best model
model.load_state_dict(torch.load(best_loss_path))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        x_batch = batch['x'].to(device)
        y_batch = batch['y'].to(device)
        outputs = model(x_batch)
        preds = torch.sigmoid(outputs).round()  # since BCEWithLogitsLoss
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# Save predictions
preds_df = pd.DataFrame({"label": all_labels, "prediction": all_preds})
preds_df.to_csv(os.path.join(RESULTS_DIR, "predictions.csv"), index=False)
print(f"Saved predictions to {RESULTS_DIR}/predictions.csv")