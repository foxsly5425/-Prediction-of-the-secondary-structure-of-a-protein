model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

sample_entry = data[10000]
sequence = sample_entry['sequence']
true_secondary_structure = sample_entry['secondary_structure']
pdb_file = sample_entry['pdb_file']
chain_id = sample_entry['chain_id']

def prepare_sequence(trigrams, word_to_idx):
    indices = [word_to_idx.get(trigram, word_to_idx['UNK']) for trigram in trigrams]
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0) 

START_TOKEN = 'X'
END_TOKEN   = 'X'
n_gramm     = 4
pad_size    = (n_gramm) // 2
padded_seq  = START_TOKEN * pad_size + sequence + END_TOKEN * pad_size
trigrams    = [padded_seq[j: j + n_gramm] for j in range(len(sequence))]

input_tensor = prepare_sequence(trigrams, word_to_idx).to(device)

with torch.no_grad():
    outputs      = model(input_tensor)  
    _, predicted = torch.max(outputs, dim=2) 
    predicted    = predicted.squeeze(0).cpu().numpy()

label_mapping = {0: 'H', 1: 'E', 2: 'L'}  

predicted_structure = [label_mapping.get(label, 'X') for label in predicted]

def visualize_prediction(sequence, true_structure, predicted_structure, pdb_file, chain_id, start=0, end=None):
    if end is None:
        end = len(sequence)

    subset_seq  = sequence[start:end]
    subset_true = true_structure[start:end]
    subset_pred = predicted_structure[start:end]

    color_map   = {'H': 'red', 'E': 'green', 'L': 'blue', 'X': 'grey'}
    true_colors = [color_map.get(char, 'grey') for char in subset_true]
    pred_colors = [color_map.get(char, 'grey') for char in subset_pred]

    plt.figure(figsize=(15, 4))

    plt.subplot(3, 1, 1)
    plt.bar(range(len(subset_true)), [1]*len(subset_true), color=true_colors, width=1)
    plt.yticks([])
    plt.title(f'Истинная вторичная структура для PDB: {pdb_file}, Chain: {chain_id}, Позиции: {start}-{end}')

    plt.subplot(3, 1, 2)
    plt.bar(range(len(subset_pred)), [1]*len(subset_pred), color=pred_colors, width=1)
    plt.yticks([])
    plt.title('Предсказанная вторичная структура')

    difference = [t != p for t, p in zip(subset_true, subset_pred)]
    diff_colors = ['yellow' if diff else 'grey' for diff in difference]
    plt.subplot(3, 1, 3)
    plt.bar(range(len(difference)), [1]*len(difference), color=diff_colors, width=1)
    plt.yticks([])
    plt.title('Ошибки предсказания (желтый - ошибка)')

    legend_elements = [Patch(facecolor=color_map[label], label=label) for label in color_map]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()

visualize_prediction(sequence, true_secondary_structure, predicted_structure, pdb_file, chain_id, start=0, end=20000)
