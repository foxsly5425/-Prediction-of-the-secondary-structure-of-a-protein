embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
nn.init.xavier_uniform_(embedding_layer.weight)
embedding_layer.weight.data[PAD_IDX] = 0
