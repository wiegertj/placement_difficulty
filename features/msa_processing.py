import math
import pandas as pd
import os
import statistics
import numpy as np
from Bio import AlignIO


def basic_msa_features(msa_filepath) -> (int, int):
    """
    Computes the number of sequences and their length in the MSA.

            Parameters:
                    msa_filepath (string): path to reference MSA

            Returns:
                    tuple: number of sequences, sequence length
    """
    alignment = AlignIO.read(msa_filepath, 'fasta')
    num_sequences = len(alignment)
    seq_length = len(alignment[0].seq)
    return num_sequences, seq_length


def gap_statistics(msa_filepath) -> (float, float):
    """
    Computes gap statistics for a reference MSA.

            Parameters:
                    msa_filepath (string): path to reference MSA

            Returns:
                    tuple: average gaps per sequence, standard deviation of gap count
    """
    alignment = AlignIO.read(msa_filepath, 'fasta')
    gap_counts = [seq.count('-') for seq in alignment]
    seq_length = len(alignment[0].seq)
    avg_gaps = statistics.mean(gap_counts) / seq_length
    std_gaps = statistics.stdev(gap_counts)

    return avg_gaps, std_gaps


def compute_entropy(msa_filepath):
    """
    Computes the per site entropies of the MSA.
    Returns summary statistics over them.

            Parameters:
                    msa_filepath (string): path to reference MSA

            Returns:
                    tuple: avg_entropy, std_entropy, min_entropy, max_entropy
    """
    alignment = AlignIO.read(msa_filepath, 'fasta')
    site_entropies = []

    num_sites = alignment.get_alignment_length()
    for site in range(num_sites):
        site_column = alignment[:, site]

        nucleotides = ['A', 'C', 'T', 'G', '-']
        ambiguous_chars = {'R': ['A', 'G'],
                           'Y': ['C', 'T'],
                           'S': ['G', 'C'],
                           'W': ['A', 'T'],
                           'K': ['G', 'T'],
                           'M': ['A', 'C'],
                           'B': ['C', 'G', 'T'],
                           'D': ['A', 'G', 'T'],
                           'H': ['A', 'C', 'T'],
                           'V': ['A', 'C', 'G']}

        # Count the occurrences of each nucleotide
        nucleotide_counts = {nucleotide: site_column.count(nucleotide) for nucleotide in nucleotides}
        total_count = sum(nucleotide_counts.values())

        probabilities = {}
        for nucleotide in nucleotides:
            count = nucleotide_counts[nucleotide]
            probabilities[nucleotide] = count / total_count

        # Assign probabilities for ambiguous characters
        for ambiguous_char, possible_nucleotides in ambiguous_chars.items():
            count = site_column.count(ambiguous_char)
            total_count += count
            for nucleotide in possible_nucleotides:
                probabilities[nucleotide] += count / total_count

            total_probability = sum(probabilities.values())
            probabilities = {nucleotide: probability / total_probability for nucleotide, probability in
                             probabilities.items()}

        entropy = 0.0
        for probability in probabilities.values():
            if probability != 0:
                entropy -= probability * math.log(probability, 2)

        site_entropies.append(entropy)

    min_entropy_ = np.min(site_entropies)
    max_entropy_ = np.max(site_entropies)
    avg_entropy_ = np.mean(site_entropies)
    std_entropy_ = np.std(site_entropies)

    return avg_entropy_, std_entropy_, min_entropy_, max_entropy_


if __name__ == '__main__':
    filenames = ["neotrop_reference.fasta", "bv_reference.fasta", "tara_reference.fasta", "13553_0_query.fasta", "21086_0_query.fasta"]
    results = []
    for file in filenames:
        filepath = os.path.join(os.pardir, "data/raw/msa", file)
        avg_gaps, std_gaps = gap_statistics(filepath)
        avg_entropy, std_entropy, min_entropy, max_entropy = compute_entropy(filepath)
        num_seq, seq_length = basic_msa_features(filepath)

        name = ""

        if file == "neotrop_reference.fasta":
            name = "neotrop"
        elif file == "bv_reference.fasta":
            name = "bv"
        elif file == "tara_reference.fasta":
            name = "tara"
        else:
            name = file.replace("_msa.fasta", "")

        results.append(
            (name, avg_gaps, std_gaps, avg_entropy, std_entropy, min_entropy, max_entropy, num_seq, seq_length))
    df = pd.DataFrame(results, columns=["dataset", "avg_gaps", "std_gaps", "avg_entropy", "std_entropy", "min_entropy",
                                        "max_entropy", "num_seq", "seq_length"])
    df.to_csv(os.path.join(os.pardir, "data/processed/features", "msa_features.csv"), index=False)
