combined_quatrograms = [quatrogram for seq in train_quatrograms + test_quatrograms for quatrogram in seq]

window_size = 4
X_max       = 100
alpha       = 0.75
total_quatrograms = len(combined_quatrograms)

print("Построение co-occurrence матрицы...")

co_pairs = []
for i, quatrogram in enumerate(tqdm(combined_quatrograms, desc="Кватрограммы")):
    quatrogram_idx = word_to_idx.get(quatrogram, word_to_idx['UNK'])
    start          = max(0, i - window_size)
    end            = min(total_quatrograms, i + window_size + 1)
    for j in range(start, end):
        if j == i:
            continue
        context_quatrogram = combined_quatrograms[j]
        context_idx        = word_to_idx.get(context_quatrogram, word_to_idx['UNK'])
        co_pairs.append((quatrogram_idx, context_idx))

print("Co-occurrence пары собраны.")

class CoOccurrenceDataset(Dataset):
    def __init__(self, co_pairs, X_max=100, alpha=0.75):
        self.pairs = Counter(co_pairs)
        self.pairs = list(self.pairs.items()) 
        self.X_max = X_max
        self.alpha = alpha

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        (i, j), count = self.pairs[idx]
        f_X_ij        = min((count / self.X_max) ** self.alpha, 1.0)
        return i, j, f_X_ij, count

glove_dataset    = CoOccurrenceDataset(co_pairs, X_max, alpha)
glove_dataloader = DataLoader(glove_dataset, batch_size=512, shuffle=True)

device           = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Используемое устройство: {device}")

word_vectors = nn.Parameter(torch.randn(vocab_size, embedding_dim, device=device) * 0.01)
word_bias    = nn.Parameter(torch.randn(vocab_size, 1, device=device) * 0.01)

optimizer    = torch.optim.AdamW([word_vectors, word_bias], lr=0.001, weight_decay=1e-4)
scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

glove_epochs = 70

print("Начало обучения GloVe...")

for epoch in range(glove_epochs):
    epoch_start_time = time.time()
    total_loss       = 0.0
    pbar             = tqdm(glove_dataloader, desc=f"Эпоха {epoch+1}/{glove_epochs}", unit="batch")

    for i, j, f_X_ij, count in pbar:
        w_i = word_vectors[i]  
        w_j = word_vectors[j]  
        b_i = word_bias[i]     
        b_j = word_bias[j]     

        pred   = torch.sum(w_i * w_j, dim=1, keepdim=True) + b_i + b_j  
        target = torch.log(torch.clamp(torch.tensor(count, dtype=torch.float32, device=device), min=1.0)).unsqueeze(1)  

        loss        = (f_X_ij.to(device).unsqueeze(1) * (pred - target) ** 2).mean()
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_([word_vectors, word_bias], max_norm=1.0)

        optimizer.step()

        pbar.set_postfix({'Loss': loss.item()})

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    avg_loss = total_loss / len(glove_dataloader)
    print(f"Epoch {epoch+1}/{glove_epochs}, Average Loss: {avg_loss:.4f}, Time: {epoch_duration:.2f} sec")

    scheduler.step(avg_loss)

embedding_matrix = word_vectors.detach().cpu().numpy() + word_bias.detach().cpu().numpy()
np.save('quatrogramm_glove_embeddings.npy', embedding_matrix)
