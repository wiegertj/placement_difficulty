import math
import multiprocessing
import statistics
import numpy as np
import pandas as pd
import os
from Bio import SeqIO
from scipy.fftpack import dct  # Discrete cosine transform
from collections import defaultdict


def dna_to_numeric(sequence):
    mapping = {'A': 63, 'C': 191, 'G': 255, 'T': 127, '-': 0, 'N': 0}
    mapping = defaultdict(lambda: 0, mapping)
    numeric_sequence = [mapping[base] for base in sequence]
    return np.array(numeric_sequence)


def encode_dna_as_image(sequence):
    width = int(math.sqrt(len(sequence)))
    height = math.ceil(len(sequence) / width)

    image = np.resize(sequence, (height, width))
    return image


def compute_hamming_distance(hash_value_1, hash_value_2):
    distance = sum(c1 != c2 for c1, c2 in zip(hash_value_1, hash_value_2))
    return distance


def compute_dct_sign_only_hash(sequence):
    numeric_sequence = dna_to_numeric(sequence)
    image = encode_dna_as_image(numeric_sequence)

    dct_coeffs = dct(dct(image, axis=0), axis=1)
    sign_only_sequence = np.sign(dct_coeffs)
    sign_only_sequence = sign_only_sequence[np.ix_(list(range(8)), list(range(8)))]
    hash_value = "".join([str(int(sign)) for sign in sign_only_sequence.flatten()])
    return hash_value


def compute_perceptual_hash_distance(msa_file):
    if msa_file == "neotrop_reference.fasta":
        query_file = msa_file.replace("_reference.fasta", "_query_10k.fasta")
    else:
        query_file = msa_file.replace("_reference.fasta", "_query.fasta")
    results = []
    counter = 0
    print(msa_file)
    for record_query in SeqIO.parse(os.path.join(os.pardir, "data/raw/query", query_file), 'fasta'):
        counter += 1
        if counter % 50 == 0:
            print(counter)
        distances = []
        hash_query = compute_dct_sign_only_hash(record_query.seq)

        for record_msa in SeqIO.parse(os.path.join(os.pardir, "data/raw/msa", msa_file), 'fasta'):
            if record_msa.id != record_query.id:
                hash_msa = compute_dct_sign_only_hash(record_msa.seq)
                distance = compute_hamming_distance(hash_msa, hash_query)

                distances.append(distance)

        max_ham = max(distances)
        min_ham = min(distances)
        avg_ham = sum(distances) / len(distances)
        std_ham = statistics.stdev(distances)

        rel_max_ham = max_ham / len(hash_query)
        rel_min_ham = min_ham / len(hash_query)
        rel_avg_ham = avg_ham / len(hash_query)
        rel_std_ham = std_ham / len(hash_query)

        name = ""

        if msa_file == "neotrop_reference.fasta":
            name = "neotrop"
        elif msa_file == "bv_reference.fasta":
            name = "bv"
        elif msa_file == "tara_reference.fasta":
            name = "tara"
        else:
            name = msa_file.replace("_msa.fasta", "")

        results.append((name, record_query.id, rel_min_ham, rel_max_ham, rel_avg_ham, rel_std_ham))
    return results, msa_file


if __name__ == '__main__':

    loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
    loo_list = loo_selection['verbose_name'].str.replace(".phy", "_reference.fasta").tolist()
    file_list = ["bv_reference.fasta", "tara_reference.fasta", "neotrop_reference.fasta"] + loo_list

    if multiprocessing.current_process().name == 'MainProcess':
        multiprocessing.freeze_support()

    pool = multiprocessing.Pool()
    results = pool.imap_unordered(compute_perceptual_hash_distance, file_list)

    for result in results:
        print("Finished processing: " + result[1] + "with query file")
        df = pd.DataFrame(result[0],
                          columns=['dataset', 'sampleId', 'min_perc_hash_ham_dist', 'max_perc_hash_ham_dist',
                                   'avg_perc_hash_ham_dist',
                                   'std_perc_hash_ham_dist'])
        df.to_csv(os.path.join(os.pardir, "data/processed/features",
                               result[1].replace("_reference.fasta", "") + "_msa_perc_hash_dist.csv"))

    pool.close()
    pool.join()

