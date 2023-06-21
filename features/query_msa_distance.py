import pandas as pd
import os
import statistics
from Bio import SeqIO


def compute_hamming_distance(msa_file, query_file) -> list:
    """
    Computes for each sequence in the query file all the hamming distances to the sequences in the MSA.
    Then for each query sequence summary statistics are computed.

            Parameters:
                    msa_file (string): path to reference MSA file
                    query_file (string): path to query file

            Returns:
                    list of: dataset, sampleId, relative min hamming dist, relative max hamming dist, relative average hamming distance, relative standard deviation hamming distance
    """
    results = []
    counter = 0
    for record_query in SeqIO.parse(os.path.join(os.pardir, "data/raw/query", query_file), 'fasta'):
        counter += 1
        if counter % 50 == 0:
            print(counter)
        distances = []
        for record_msa in SeqIO.parse(os.path.join(os.pardir, "data/raw/msa", msa_file), 'fasta'):
            if record_msa.id != record_query.id:
                distance = sum(ch1 != ch2 for ch1, ch2 in zip(record_query.seq, record_msa.seq))
                distances.append(distance)
            else:
                print(record_query.id)

        max_ham = max(distances)
        min_ham = min(distances)
        avg_ham = sum(distances) / len(distances)
        std_ham = statistics.stdev(distances)

        rel_max_ham = max_ham / len(record_query.seq)
        rel_min_ham = min_ham / len(record_query.seq)
        rel_avg_ham = avg_ham / len(record_query.seq)
        rel_std_ham = std_ham / len(record_query.seq)

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

    return results


if __name__ == '__main__':

    for msa_file, query_file in [("13553_0_reference.fasta", "13553_0_query.fasta"),
                                 ("21086_0_reference.fasta", "21086_0_query.fasta"),
                                 ("neotrop_reference", "neutrop_query_10k.fasta"),
                                 ("bv_reference", "bv_query.fasta"), ("tara_reference", "tara_query.fasta")]:
        result_tmp = compute_hamming_distance(msa_file, query_file)

        df = pd.DataFrame(result_tmp, columns=['dataset', 'sampleId', 'min_ham_dist', 'max_ham_dist', 'avg_ham_dist',
                                               'std_ham_dist'])
        df.to_csv(os.path.join(os.pardir, "data/processed/features",
                               msa_file.replace("_reference.fasta", "") + "_msa_dist.csv"))
