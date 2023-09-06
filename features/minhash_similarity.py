import random

import sourmash
import os
import time
import types
import pandas as pd
import statistics
import multiprocessing
from Bio import SeqIO, AlignIO
from scipy.stats import skew, kurtosis

module_path = os.path.join(os.pardir, "configs/feature_config.py")
feature_config = types.ModuleType('feature_config')
feature_config.__file__ = module_path

with open(module_path, 'rb') as module_file:
    code = compile(module_file.read(), module_path, 'exec')
    exec(code, feature_config.__dict__)
nucleotide_ambiguity_code = {
    'R': ['A', 'G'],
    'Y': ['C', 'T'],
    'S': ['G', 'C'],
    'W': ['A', 'T'],
    'K': ['G', 'T'],
    'M': ['A', 'C'],
    'B': ['C', 'G', 'T'],
    'D': ['A', 'G', 'T'],
    'H': ['A', 'C', 'T'],
    'V': ['A', 'C', 'G'],
    'N': ['A', 'C', 'G', 'T']
}

def replace_ambiguity_chars(char):
    if char in nucleotide_ambiguity_code:
        return random.choice(nucleotide_ambiguity_code[char])
    else:
        return char


def multiprocess_minhash_similarity(query_filename, msa_filename):
    data = []
    msa = AlignIO.read(os.path.join(os.pardir, "data/raw/msa", msa_filename), 'fasta')
    results = []

    for query_record in SeqIO.parse(os.path.join(os.pardir, "data/raw/query", query_filename), 'fasta'):
        query_jacc_sims25 = []
        query_jacc_sims50 = []
        for seq_record in msa:
            seq_rec = ''.join(map(replace_ambiguity_chars, seq_record.seq)).replace("-","")
            query_seq = ''.join(map(replace_ambiguity_chars, query_record.seq)).replace("-","")
            mh1 = sourmash.MinHash(n=0, ksize=25, scaled=1)
            mh1.add_sequence(seq_rec.replace("-",""))
            mh2 = sourmash.MinHash(n=0, ksize=25, scaled=1)
            mh2.add_sequence(query_seq.replace("-",""))
            query_jacc_sim25 = round(mh1.jaccard(mh2), 2)
            mh1 = sourmash.MinHash(n=0, ksize=50, scaled=1)
            mh1.add_sequence(seq_rec.replace("-", ""))
            mh2 = sourmash.MinHash(n=0, ksize=50, scaled=1)
            mh2.add_sequence(query_seq.replace("-", ""))
            query_jacc_sim50 = round(mh1.jaccard(mh2), 2)
            query_jacc_sims25.append(query_jacc_sim25)
            query_jacc_sims50.append(query_jacc_sim50)
        mean25 = statistics.mean(query_jacc_sims25)
        mean50 = statistics.mean(query_jacc_sims50)
        std25 = statistics.stdev(query_jacc_sims25)
        std50 = statistics.stdev(query_jacc_sims50)
        min25 = min(query_jacc_sims25)
        min50 = min(query_jacc_sims50)
        max50 = max(query_jacc_sims50)
        max25 = max(query_jacc_sims25)
        results.append((msa_filename, query_record.id, mean25, mean50, std25, std50, min25, min50, max25, max50))
    return results


if __name__ == '__main__':

    module_path = os.path.join(os.pardir, "configs/feature_config.py")

    feature_config = types.ModuleType('feature_config')
    feature_config.__file__ = module_path

    with open(module_path, 'rb') as module_file:
        code = compile(module_file.read(), module_path, 'exec')
        exec(code, feature_config.__dict__)

    loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
    loo_selection = loo_selection.drop_duplicates(subset=["verbose_name"], keep="first")
    filenames = loo_selection['verbose_name'].str.replace(".phy", "_reference.fasta").tolist()

    if multiprocessing.current_process().name == 'MainProcess':
        multiprocessing.freeze_support()

    counter_msa = 0
    for msa_file in filenames:
        print(str(counter_msa) + "/" + str(len(filenames)))
        counter_msa += 1
        print("started: " + msa_file)
        if msa_file == "neotrop_reference.fasta":
            query_file = msa_file.replace("reference.fasta", "query_10k.fasta")
        else:
            query_file = msa_file.replace("reference.fasta", "query.fasta")

        # Skip already processed
        potential_path = os.path.join(os.pardir, "data/processed/features",
                                   msa_file.replace("_reference.fasta", "") + "_minhash_0".replace("0.",
                                                                                        "") + "_1000" + ".csv")
        if os.path.exists(potential_path):
            print("Skipped: " + msa_file + " already processed")
            continue

        results = []

        bloom_filters_MSA = []
        string_kernel_features = []

        if os.path.exists(os.path.join(os.pardir, "data/raw/query", query_file)):
            no_queries = len(list(SeqIO.parse(os.path.join(os.pardir, "data/raw/query", query_file), 'fasta').records))
        else:
            print("Query file not found: " + os.path.join(os.pardir, "data/raw/query", query_file))
            continue

        interval_start = feature_config.KMER_PROCESSING_INTERVAL_START  # sequence number to start with in query (last file number)
        bound = feature_config.KMER_PROCESSING_COUNT  # how many sequences
        num_sequences = sum(1 for _ in SeqIO.parse(os.path.join(os.pardir, "data/raw/msa", msa_file), 'fasta'))

        while True:
            interval_start += feature_config.KMER_PROCESSING_STEPSIZE
            result_tmp = multiprocess_minhash_similarity(query_file, msa_file)

            results.extend(result_tmp)
            df = pd.DataFrame(results,
                              columns=['dataset', 'sampleId', 'minhash_mean_25', 'minhash_mean_50', 'minhash_std_25', 'minhash_std_50',
                                       'minhash_min_25', 'minhash_min_50', 'minhash_max_25', "minhash_max_50"])
            df.to_csv(os.path.join(os.pardir, "data/processed/features",
                                   msa_file.replace("_reference.fasta", "") + "_minhash_0".replace("0.",
                                                                                        "") + "_" + str(
                                       interval_start) + ".csv"), index=False)
            results = []

            if interval_start >= no_queries or interval_start >= bound:  # stop if we reached bound or no query samples left
                break
