import os
import json
import warnings

from concurrent.futures    import ThreadPoolExecutor, as_completed
from Bio.PDB               import PDBParser, DSSP
from Bio.SeqUtils          import seq1
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from tqdm                  import tqdm 

warnings.simplefilter('ignore', PDBConstructionWarning)

def extract_secondary_structure_from_pdb_file(pdb_file, pdb_dir, dssp_exe_path):
    try:
        pdb_path = os.path.join(pdb_dir, pdb_file)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_file[:-4], pdb_path)
        model = structure[0]

        dssp = DSSP(model, pdb_path, dssp=dssp_exe_path)

        ss_dict = {}
        for key in dssp.keys():
            chain_id, res_id, icode = key[0], key[1][1], key[1][2].strip()
            if chain_id not in ss_dict:
                ss_dict[chain_id] = []
            sec_struct = dssp[key][2]
            if sec_struct == "H":
                ss_dict[chain_id].append("H")
            elif sec_struct == "E":
                ss_dict[chain_id].append("E")
            else:
                ss_dict[chain_id].append("L")

        return pdb_file, ss_dict
    except Exception as e:
        print(f"Ошибка при обработке вторичной структуры {pdb_file}: {e}")
        return pdb_file, None

def extract_secondary_structure_concurrently(pdb_files, pdb_dir, dssp_exe_path, max_workers=16):
    pdb_secondary_structures = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdb = {executor.submit(extract_secondary_structure_from_pdb_file, pdb_file, pdb_dir, dssp_exe_path): pdb_file for pdb_file in pdb_files}

        for future in tqdm(as_completed(future_to_pdb), total=len(future_to_pdb), desc="Извлечение вторичной структуры"):
            pdb_file, ss_dict = future.result()
            if ss_dict:
                pdb_secondary_structures[pdb_file] = ss_dict

    return pdb_secondary_structures


def get_amino_acid_sequences_per_chain(pdb_file, pdb_dir):
    try:
        pdb_path = os.path.join(pdb_dir, pdb_file)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_file[:-4], pdb_path)
    except Exception as e:
        print(f"Ошибка при парсинге структуры {pdb_file}: {e}")
        return pdb_file, None

    sequences = {}
    for model in structure:
        for chain in model:
            chain_id = chain.id
            chain_sequence = []
            for residue in chain:
                if residue.get_id()[0] == ' ':  
                    resname = residue.get_resname()
                    try:
                        aa = seq1(resname)
                        chain_sequence.append(aa)
                    except KeyError:
                        chain_sequence.append('X') 
            if chain_sequence:
                sequences[chain_id] = ''.join(chain_sequence)
    return pdb_file, sequences

def get_sequences_from_pdb_files_concurrently(pdb_files, pdb_dir, max_workers=16):
    sequences = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdb = {executor.submit(get_amino_acid_sequences_per_chain, pdb_file, pdb_dir): pdb_file for pdb_file in pdb_files}

        for future in tqdm(as_completed(future_to_pdb), total=len(future_to_pdb), desc="Извлечение последовательностей"):
            pdb_file, chain_sequences = future.result()
            if chain_sequences:
                sequences[pdb_file] = chain_sequences

    return sequences

def filter_and_collect_results(sequences, secondary_structures):
    result = []
    for pdb_file, chains in sequences.items():
        if pdb_file in secondary_structures:
            ss_chains = secondary_structures[pdb_file]
            for chain_id, sequence in chains.items():
                if chain_id in ss_chains:
                    secondary = ''.join(ss_chains[chain_id])
                    if len(sequence) == len(secondary):
                        result.append({
                            "pdb_file": pdb_file,
                            "chain_id": chain_id,
                            "sequence": sequence,
                            "secondary_structure": secondary
                        })
                    else:
                        print(f"Несоответствие длины для {pdb_file} цепь {chain_id}: длина последовательности {len(sequence)}, длина структуры {len(secondary)}")
    return result

def save_results_to_file(result, output_file):
    with open(output_file, 'w') as file:
        json.dump(result, file, indent=4)
    print(f"Результаты сохранены в {output_file}")


def main():
    dssp_exe_path = r'C:\Program Files\DSSP\bin\mkdssp.exe'  
    if not os.path.exists(dssp_exe_path):
        print(f"Не удалось найти DSSP по пути: {dssp_exe_path}")
        return

    pdb_dir = './pdb_files'
    if not os.path.exists(pdb_dir):
        print(f"Папка {pdb_dir} не найдена.")
        return

    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]

    print(f"Найдено {len(pdb_files)} PDB-файлов для обработки.")

    if not pdb_files:
        print("Нет PDB-файлов для обработки.")
        return

    print("Начало извлечения вторичной структуры...")
    pdb_secondary_structures = extract_secondary_structure_concurrently(pdb_files, pdb_dir, dssp_exe_path, max_workers=16)  
    print(f"Извлечено вторичных структур для файлов: {len(pdb_secondary_structures)}")

    print("Начало извлечения аминокислотных последовательностей...")
    sequences = get_sequences_from_pdb_files_concurrently(pdb_files, pdb_dir, max_workers=16)  
    print(f"Извлечено последовательностей для файлов: {len(sequences)}")

    print("Начало фильтрации и сбора результатов...")
    result = filter_and_collect_results(sequences, pdb_secondary_structures)
    print(f"Количество корректных записей: {len(result)}")

    output_file = 'amin_seq_second_struct.json' 
    save_results_to_file(result, output_file)

if __name__ == "__main__":
    main()
