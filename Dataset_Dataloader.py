class ProteinDatasetForModel(Dataset):
    def __init__(self, trigrams, labels, word_to_idx, n_gramm=3, pad_size=1):
        self.trigrams    = trigrams
        self.labels      = labels
        self.word_to_idx = word_to_idx
        self.n_gramm     = n_gramm
        self.pad_size    = pad_size
        self.START_TOKEN = 'X'
        self.END_TOKEN   = 'X'
        self.preprocess()

    def preprocess(self):
        self.indexed_trigrams = []
        self.indexed_labels  = []
        for trigrams, labels in zip(self.trigrams, self.labels):
            indices = [self.word_to_idx.get(trigram, self.word_to_idx['UNK']) for trigram in trigrams]
            self.indexed_trigrams.append(indices)
            self.indexed_labels.append(labels)

    def __len__(self):
        return len(self.indexed_trigrams)

    def __getitem__(self, idx):
        return self.indexed_trigrams[idx], self.indexed_labels[idx]


train_dataset_model = ProteinDatasetForModel(train_quatrograms, train_labels, word_to_idx)
test_dataset_model  = ProteinDatasetForModel(test_quatrograms, test_labels, word_to_idx)

def custom_collate_fn(batch):

    trigrams, labels = zip(*batch)

    max_len          = max(len(seq) for seq in trigrams)

    PAD_TOKEN = 'PAD'
    PAD_IDX   = word_to_idx[PAD_TOKEN]

    padded_trigrams = []
    padded_labels   = []
    for trigram_seq, label_seq in zip(trigrams, labels):
        pad_length = max_len - len(trigram_seq)
        padded_trigrams.append(trigram_seq + [PAD_IDX] * pad_length)
        padded_labels.append(label_seq + [-1] * pad_length)

    trigrams_tensor = torch.tensor(padded_trigrams, dtype=torch.long)
    labels_tensor   = torch.tensor(padded_labels, dtype=torch.long)

    return trigrams_tensor, labels_tensor

train_loader = DataLoader(
    train_dataset_model,
    batch_size=128,
    shuffle=True,
    pin_memory=False, 
    collate_fn=custom_collate_fn
)

test_loader = DataLoader(
    test_dataset_model,
    batch_size=128,
    shuffle=False,
    pin_memory=False,  
    collate_fn=custom_collate_fn
)
