import os
import gzip
import shutil
import pandas as pd
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, DatasetDict
import torch
# from torch.utils.data import Dataset
from tokenizers import my_tokenizer_word, my_tokenizer_number

LOAD_DATA = 'LOCAL'
LOAD_DATA = 'HUG'

DATA_DIR = './data'
FILE_NAME = 'movie_data.csv'
FILE_NAME_GZ = 'movie_data.csv.gz'

LOAD_DATA_PATH = os.path.join(DATA_DIR, FILE_NAME)
LOAD_DATA_PATH_GZ = os.path.join(DATA_DIR, FILE_NAME_GZ)

def load_data_as_df(source="LOCAL", test_size=0.2, random_state=42):
    """
    Load dataset from LOCAL files or Hugging Face.
    Returns: train_df, test_df
    """

    if source == "LOCAL":
        # --- Load local file ---
        if os.path.exists(LOAD_DATA_PATH):
            df = pd.read_csv(LOAD_DATA_PATH)

        elif os.path.exists(LOAD_DATA_PATH_GZ):
            with gzip.open(LOAD_DATA_PATH_GZ, "rb") as f_in:
                with open(LOAD_DATA_PATH, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            df = pd.read_csv(LOAD_DATA_PATH)

        else:
            raise FileNotFoundError(
                f"No dataset found in {DATA_DIR}. Expected {FILE_NAME} or {FILE_NAME_GZ}"
            )

        # --- Rename columns to match Hugging Face schema ---
        column_map = {
            "review": "text",
            "sentiment": "label"
        }
        df = df.rename(columns=column_map)

        # --- Validate schema ---
        required_cols = {"text", "label"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Dataset must contain columns {required_cols}, got {df.columns}")

        # --- Split to train/test ---
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )

        return train_df, test_df

    elif source == "HUG":
        # --- Load from Hugging Face ---
        dataset = load_dataset("imdb")

        train_df = dataset["train"].to_pandas()
        test_df = dataset["test"].to_pandas()

        return train_df, test_df

    else:
        raise ValueError("source must be either 'LOCAL' or 'HUG'")
    
    
def load_data(source="LOCAL", test_size=0.2, random_state=42):
    """
    Load dataset from LOCAL files or Hugging Face.
    Always returns a DatasetDict with 'train' and 'test'
    and columns: ['text', 'label'].
    """

    if source == "LOCAL":
        # --- Load local file ---
        if os.path.exists(LOAD_DATA_PATH):
            df = pd.read_csv(LOAD_DATA_PATH)

        elif os.path.exists(LOAD_DATA_PATH_GZ):
            with gzip.open(LOAD_DATA_PATH_GZ, "rb") as f_in:
                with open(LOAD_DATA_PATH, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            df = pd.read_csv(LOAD_DATA_PATH)

        else:
            raise FileNotFoundError(
                f"No dataset found in {LOAD_DATA_PATH}. Expected {FILE_NAME} or {FILE_NAME_GZ}"
            )

        # --- Rename columns to match Hugging Face schema ---
        column_map = {
            "review": "text",
            "sentiment": "label"
        }
        df = df.rename(columns=column_map)

        # --- Validate schema ---
        required_cols = {"text", "label"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Dataset must contain columns {required_cols}, got {df.columns}")

        # --- Split ---
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )

        train_df, valid_df = train_test_split(
            train_df, test_size=max(0.25, test_size), random_state=random_state
        )

        train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
        test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))
        valid_ds = Dataset.from_pandas(valid_df.reset_index(drop=True))
        

        return DatasetDict({
            "train": train_ds,
            "test": test_ds,
            "validation": valid_ds
        })

    elif source == "HUG":
        dataset = load_dataset("imdb")
        return dataset

    else:
        raise ValueError("source must be either 'LOCAL' or 'HUG'")
    
    
class TextClassificationDataset_wasted_time(Dataset):
    def __init__(self, dataset_text, dataset_label, vocab, max_len=200):
        """
        dataset: Hugging Face Dataset format (train or test split)
        vocab: dictionary mapping word -> index
        max_len: max sequence length
        """
        self.dataset_text = dataset_text
        self.dataset_label = dataset_label
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']  # or 'review' depending on your column
        label = item['label']  # or 'sentiment'

        # tokenize to numbers
        # indices = [my_tokenizer_number(t, self.vocab, max_len=self.max_len) for t in item["text"]]
        indices = my_tokenizer_number(text, self.vocab, self.max_len)

        # convert to tensor
        x = torch.tensor(indices, dtype=torch.long)
        y = torch.tensor(label, dtype=torch.long)
        num_of_token = len(x)

        # optionally pad if shorter than max_len
        if len(x) < self.max_len:
            pad_len = self.max_len - len(x)
            pad_idx = self.vocab['<pad>']
            x = torch.cat([x, torch.full((pad_len,), pad_idx, dtype=torch.long)])

        data = {
            'tokenized_text': x,
            'label': y.float(),
            'num_of_token': num_of_token
        }
        # return x, y.float(), length_of_token
        return data
    
# new and performance improved
class TextClassificationDataset(Dataset):
    def __init__(self, dataset_text_indices, dataset_label, vocab, max_len=200):
        """
        dataset: Hugging Face Dataset format (train or test split)
        vocab: dictionary mapping word -> index
        max_len: max sequence length
        """
        self.dataset_text_indices = dataset_text_indices
        self.dataset_label = dataset_label
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset_text_indices)

    def __getitem__(self, idx):
        pad_idx = self.vocab['<pad>']
        if isinstance(idx, list):
            text_indices = [None] * len(idx)
            for i, i_idx in enumerate(idx):
                text_indices[i] = self.dataset_text_indices[i_idx]
                if len(text_indices[i]) < self.max_len:
                    text_indices[i].extend([pad_idx] *\
                        (self.max_len - len(text_indices[i]))) 
                else:
                    text_indices[i] = text_indices[i][:self.max_len]
            label = [self.dataset_label[i] for i in idx]
        elif isinstance(idx, int):
            text_indices = self.dataset_text_indices[idx]
            # optionally pad if shorter than max_len
            if len(text_indices) < self.max_len:
                pad_len = self.max_len - len(text_indices)
                # pad_idx = self.vocab['<pad>']
                text_indices.extend([pad_idx] * pad_len)
            
            label = self.dataset_label[idx]
        
        
        
        # text_indices = self.dataset_text_indices[idx]
        # label = self.dataset_label[idx]
        # text is already tokenized to numbers and truncated to max_len in the collate_fn

        # convert to tensor
        x = torch.tensor(text_indices, dtype=torch.long)
        y = torch.tensor(label, dtype=torch.float32)
        num_of_token = torch.tensor(len(x))

        data = {
            'x': x,
            'y': y
            # 'num_of_token': num_of_token
        }
        # return x, y.float(), length_of_token
        return data

# Collate function
# def collate_fn(batch):
#     x = [item['x'] for item in batch]
#     y = [item['y'] for item in batch]
#     num_of_token = [item['num_of_token'] for item in batch]

#     indices_padded = torch.nn.utils.rnn.pad_sequence(
#         x, batch_first=True, padding_value=0  # Assuming <pad> = 0
#     )

#     return {
#         'x': indices_padded,
#         'y': torch.tensor(y, dtype=torch.float32),
#         'num_of_token': torch.tensor(num_of_token, dtype=torch.long),
#     }
    
def collate_fn(batch):
    x = [item['x'] for item in batch]
    y = [item['y'] for item in batch]
    num_of_token = [item['num_of_token'] for item in batch]

    x_padded = torch.nn.utils.rnn.pad_sequence(
        x, batch_first=True, padding_value=0  # Assuming <pad> = 0
    )

    return {
        'x': x_padded,
        'y': torch.stack(y),
        'num_of_token': torch.tensor(num_of_token),
    }
        

def build_vocab(dataset, text_column='text', vocab_size=25000, min_freq=2, specials=['<pad>', '<unk>']):
    """
    Build a vocabulary dictionary from a Hugging Face dataset.
    
    Args:
        dataset: Hugging Face Dataset (e.g., dataset['train'])
        text_column (str): Name of the text column in the dataset
        vocab_size (int): Maximum number of words to include in vocab (including specials)
        min_freq (int): Minimum frequency for a word to be included
        specials (list): List of special tokens (like <pad>, <unk>)
        
    Returns:
        vocab (dict): Mapping word -> index
    """
    word_counts = Counter()
    
    # Count tokens in the dataset
    for example in dataset:
        tokens = my_tokenizer_word(example[text_column])
        word_counts.update(tokens)
    
    # Initialize vocab with special tokens
    vocab = {token: idx for idx, token in enumerate(specials)}
    num_specials = len(vocab)
    
    # Get most common words respecting min_freq
    filtered_words = [(word, count) for word, count in word_counts.most_common()
                      if count >= min_freq]
    
    # Limit to vocab_size
    filtered_words = filtered_words[:vocab_size - num_specials]
    
    # Add to vocab
    offset = num_specials
    for word, _ in filtered_words:
        vocab[word] = offset
        offset += 1
    
    print(f"Vocabulary size: {len(vocab)} (including specials)")
    print("Top 10 words:", filtered_words[:10])
    
    return vocab