import math
import multiprocessing
import statistics
import types
import numpy as np
import pandas as pd
import os
from Bio import SeqIO
from scipy.fftpack import dct  # Discrete cosine transformation
from collections import defaultdict

from scipy.stats import kurtosis, skew


def dna_to_numeric(sequence):
    mapping = {'A': 42, 'C': 95, 'G': 137, 'T': 180, '-': 222, 'N': 255}
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
    try:
        sign_only_sequence = sign_only_sequence[np.ix_(list(range(feature_config.SIGN_ONLY_MATRIX_SIZE)),
                                                       list(range(feature_config.SIGN_ONLY_MATRIX_SIZE)))]
        hash_value = "".join([str(int(sign)) for sign in sign_only_sequence.flatten()])
    except IndexError:
        print("image too small, skipped")
        return 0
    return hash_value


def compute_perceptual_hash_distance(msa_file):
    if msa_file == "neotrop_reference.fasta":
        query_file = msa_file.replace("_reference.fasta", "_query_10k.fasta")
    else:
        query_file = msa_file.replace("_reference.fasta", "_query.fasta")
    results = []
    counter = 0
    print(msa_file)
    # Skip already processed
    potential_path = os.path.join(os.pardir, "data/processed/features",
                                  msa_file.replace("_reference.fasta", "") + "16n_msa_perc_hash_dist" + ".csv")
    if os.path.exists(potential_path):
        print("Skipped: " + msa_file + " already processed")
        return 0

    for record_query in SeqIO.parse(os.path.join(os.pardir, "data/raw/query", query_file), 'fasta'):
        counter += 1
        if counter % 50 == 0:
            print(counter)
        distances = []
        hash_query = compute_dct_sign_only_hash(record_query.seq)

        for record_msa in SeqIO.parse(os.path.join(os.pardir, "data/raw/msa", msa_file), 'fasta'):
            if record_msa.id != record_query.id:
                hash_msa = compute_dct_sign_only_hash(record_msa.seq)
                if hash_msa != 0:
                    distance = compute_hamming_distance(hash_msa, hash_query)
                    distances.append(distance)
                else:
                    return 0

        max_ham = max(distances)
        min_ham = min(distances)
        avg_ham = sum(distances) / len(distances)
        std_ham = statistics.stdev(distances)

        sk_ham = skew(distances)
        kur_ham = kurtosis(distances, fisher=False)

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

        results.append((name, record_query.id, rel_min_ham, rel_max_ham, rel_avg_ham, rel_std_ham, sk_ham, kur_ham))
    return results, msa_file


if __name__ == '__main__':

    module_path = os.path.join(os.pardir, "configs/feature_config.py")

    feature_config = types.ModuleType('feature_config')
    feature_config.__file__ = module_path

    with open(module_path, 'rb') as module_file:
        code = compile(module_file.read(), module_path, 'exec')
        exec(code, feature_config.__dict__)

    loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
    filenames = loo_selection['verbose_name'].str.replace(".phy", "_reference.fasta").tolist()

    if feature_config.INCUDE_TARA_BV_NEO:
        filenames = filenames + ["bv_reference.fasta", "neotrop_reference.fasta", "tara_reference.fasta"]

    for file in filenames:
        if len(next(SeqIO.parse(os.path.join(os.pardir, "data/raw/msa", file), 'fasta').records).seq) > 15000:
            filenames.remove(file)

    if multiprocessing.current_process().name == 'MainProcess':
        multiprocessing.freeze_support()

    pool = multiprocessing.Pool()
    results = pool.imap_unordered(compute_perceptual_hash_distance, filenames)

    for result in results:
        if result != 0:
            print("Finished processing: " + result[1] + "with query file")
            df = pd.DataFrame(result[0],
                              columns=['dataset', 'sampleId', 'min_perc_hash_ham_dist', 'max_perc_hash_ham_dist',
                                       'avg_perc_hash_ham_dist',
                                       'std_perc_hash_ham_dist', 'skewness_perc_hash_ham_dist', 'kurtosis_perc_hash_ham_dist'])
            df.to_csv(os.path.join(os.pardir, "data/processed/features",
                                   result[1].replace("_reference.fasta", "") + str(
                                       feature_config.SIGN_ONLY_MATRIX_SIZE) + "n_msa_perc_hash_dist.csv"))

    pool.close()
    pool.join()
