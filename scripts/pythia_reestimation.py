import os

import tarfile

import subprocess
import pandas as pd

from Bio import AlignIO, SeqIO

difficulties_path = os.path.join(os.pardir, "data/treebase_difficulty.csv")
difficulties_df = pd.read_csv(difficulties_path, index_col=False, usecols=lambda column: column != 'Unnamed: 0')
difficulties_df.drop_duplicates(subset=["verbose_name"], keep="first", inplace=True)
results = []
counter = 0
try:
    current_processed = pd.read_csv(os.path.join(os.pardir, "data/treebase_difficulty_new.csv"))
    current_processed_names = current_processed["name"].values.tolist()
except FileNotFoundError:
    current_processed_names = []
print(current_processed_names)
for index, row in difficulties_df.iterrows():
    counter +=1
    print(counter)
    duplicate_counter = 0
    file = row["verbose_name"]
    if file in current_processed_names:
        print("already processed")
        continue
    dtype = str(row["data_type"])
    if (dtype == "DNA") or (dtype == "DataType.DNA"):
        datatype = "DNA"
    else:
        datatype == "AA"
    file_path = os.path.join(os.pardir, "data/TreeBASEMirror-main/trees/" + file)
    tar_file = os.path.join(file_path, file + ".tar.gz")
    print(tar_file)

    try:
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(file_path)
    except FileNotFoundError:
        print("Not found: " + file + " skipped")
        continue

    extracted_path = os.path.join(file_path,
                                  'msa.fasta')

    MSA = AlignIO.read(extracted_path, 'fasta')

    unique_sequences = set()
    alignment_dedup = []
    for record in MSA:
        # Convert the sequence to a string for comparison
        sequence_str = str(record.seq)

        # Check if the sequence is unique
        if sequence_str not in unique_sequences:
            unique_sequences.add(sequence_str)
            alignment_dedup.append(record)
        else:
            duplicate_counter += 1
    print("Duplicate counter: " + str(duplicate_counter))

    new_name_msa = os.path.join(file_path, file.replace(".phy", "_reference_dedup.fasta"))

    SeqIO.write(alignment_dedup, new_name_msa, "fasta")

    raxml_path = subprocess.check_output(["which", "raxml-ng"], text=True).strip()
    command = ["pythia", "--msa", new_name_msa,
               "--raxmlng", raxml_path]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        pythia_output = result.stdout
        pythia_output = result.stdout
        pythia_error = result.stderr  # Capture stderr
        print(pythia_error)
    except subprocess.CalledProcessError:
        print("Error occured")
        continue

    is_index = result.stderr.find("is: ")
    if is_index != -1:
        value_string = result.stderr[is_index + 4:is_index + 8]  # Add 4 to skip "is: "
        extracted_value = float(value_string)
    else:
        print("No match found")

    last_float_before = extracted_value
    results.append((file, datatype, last_float_before, duplicate_counter))

    res_df = pd.DataFrame(results, columns=["name", "datatype", "difficulty", "no_duplicates"])
    results = []

    if not os.path.isfile(os.path.join(os.pardir, "data/treebase_difficulty_new.csv")):
        res_df.to_csv(os.path.join(os.pardir, "data/treebase_difficulty_new.csv"),
                     index=False, header=True)
    else:
        res_df.to_csv(os.path.join(os.pardir, "data/treebase_difficulty_new.csv"),
                     index=False,
                     mode='a', header=False)

