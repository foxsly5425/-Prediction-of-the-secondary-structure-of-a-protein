import warnings
warnings.filterwarnings("ignore")

import json
import math
import time
import numpy               as np
import torch
import torch.nn            as nn
import torch.nn.functional as F
import matplotlib.pyplot   as plt

import wandb

from sklearn.model_selection import train_test_split
from sklearn.metrics         import f1_score, precision_score, recall_score
from torch.utils.data        import Dataset, DataLoader
from collections             import Counter
from tqdm                    import tqdm

from transformers            import BertModel, BertTokenizer


START_TOKEN = 'X'
END_TOKEN   = 'X'
n_gramm     = 4
pad_size    = (n_gramm) // 2
MIN_COUNT   = 3  

with open('amin_seq_second_struct.json', 'r') as file:
    data = json.load(file)

sequences  = [entry['sequence'] for entry in data]
structures = [entry['secondary_structure'] for entry in data]

# Гене
n_gramm_seq = []
struct_seq = []
for seq, struct in zip(sequences, structures):
    padded_seq = START_TOKEN * pad_size + seq + END_TOKEN * pad_size
    quatrograms = [padded_seq[j: j + n_gramm] for j in range(len(seq))]
    if len(quatrograms) != len(struct):
        continue
    n_gramm_seq.append(quatrograms)
    struct_seq.append(struct)

print(f"Всего последовательностей после фильтрации: {len(n_gramm_seq)}")

full_quatrograms     = [quatrogram for seq in n_gramm_seq for quatrogram in seq]

quatrogram_counts    = Counter(full_quatrograms)

filtered_quatrograms = [quatrogram if quatrogram_counts[quatrogram] >= MIN_COUNT else 'UNK' for quatrogram in full_quatrograms]

vocab         = sorted(set(filtered_quatrograms))
vocab_size    = len(vocab)
embedding_dim = 256
print(f"Размер словаря после фильтрации: {vocab_size}")

idx_to_word = {i: vocab[i] for i in range(vocab_size)}
word_to_idx = {vocab[i]: i for i in range(vocab_size)}

if 'UNK' not in word_to_idx:
    word_to_idx['UNK']            = len(word_to_idx)
    idx_to_word[len(idx_to_word)] = 'UNK'
    vocab_size += 1
    filtered_quatrograms          = [quatrogram if quatrogram_counts.get(quatrogram, 0) >= MIN_COUNT else 'UNK' for quatrogram in full_quatrograms]

print(f"Итоговый размер словаря с 'UNK': {vocab_size}")

PAD_TOKEN                     = 'PAD'
word_to_idx[PAD_TOKEN]        = len(word_to_idx)  
idx_to_word[len(idx_to_word)] = PAD_TOKEN
vocab_size += 1

PAD_IDX = word_to_idx[PAD_TOKEN] 

padding_vector   = np.zeros((1, embedding_dim), dtype=np.float32)
embedding_matrix = np.random.randn(vocab_size -1, embedding_dim).astype(np.float32) * 0.01 
embedding_matrix = np.vstack([embedding_matrix, padding_vector])  

print(f"PAD_IDX: {PAD_IDX}, num_embeddings: {embedding_matrix.shape[0]}")
assert PAD_IDX < embedding_matrix.shape[0], 

with open('word_to_idx.json', 'w') as f:
    json.dump(word_to_idx, f)

all_labels = []
for struct in struct_seq:
    label_mapping  = {'H': 0, 'E': 1, 'L': 2}  
    numeric_struct = [label_mapping.get(char, -1) for char in struct]
    all_labels.append(numeric_struct)

test_size = 0.15
random_seed = 42

train_quatrograms, test_quatrograms, train_labels, test_labels = train_test_split(
    n_gramm_seq,
    all_labels,
    test_size=test_size,
    random_state=random_seed
)

print(f"Размер обучающей выборки: {len(train_quatrograms)}")
print(f"Размер тестовой выборки : {len(test_quatrograms) }")
