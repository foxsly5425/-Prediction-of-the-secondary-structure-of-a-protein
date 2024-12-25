tokenizer  = BertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)
bert_model = BertModel.from_pretrained('Rostlab/prot_bert').to('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.eval()

def get_bert_embedding(trigram, tokenizer, model, device):
    tokens = list(trigram)
    tokens = ' '.join(tokens)  
    inputs = tokenizer(tokens, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  
    return cls_embedding.squeeze(0).cpu().numpy()


embedding_dim    = bert_model.config.hidden_size  
embedding_matrix = np.random.randn(vocab_size, embedding_dim).astype(np.float32) * 0.01 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Используемое устройство для BERT: {device}")

for idx, trigram in tqdm(idx_to_word.items(), desc="Генерация эмбеддингов с помощью BERT"):
    if trigram == PAD_TOKEN:
        embedding_matrix[idx] = np.zeros(embedding_dim, dtype=np.float32)  
    elif trigram == 'UNK':
        embedding_matrix[idx] = np.random.randn(embedding_dim).astype(np.float32) * 0.01 
    else:
        embedding_matrix[idx] = get_bert_embedding(trigram, tokenizer, bert_model, device)

np.save('quatrogramm_prot_bert_embeddings.npy', embedding_matrix)
