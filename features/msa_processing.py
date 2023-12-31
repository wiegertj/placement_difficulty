import math
import random
import types
import Bio
import pandas as pd
import os
import statistics
import numpy as np
from Bio import AlignIO
from Bio.Align import AlignInfo
from scipy.stats import skew, kurtosis
from uncertainty.ApproximateEntropy import ApproximateEntropy
from uncertainty.Complexity import ComplexityTest
from uncertainty.CumulativeSum import CumulativeSums
from uncertainty.FrequencyTest import FrequencyTest
from uncertainty.Matrix import Matrix
from uncertainty.RandomExcursions import RandomExcursions
from uncertainty.RunTest import RunTest
from uncertainty.Serial import Serial
from uncertainty.Spectral import SpectralTest


def basic_msa_features(msa_filepath) -> (int, int):
    """
    Computes the number of sequences and their length in the MSA.

            Parameters:
                    :param msa_filepath: path to reference MSA

            Returns:
                    :return tuple: number of sequences, sequence length
    """
    alignment = AlignIO.read(msa_filepath, 'fasta')
    num_sequences = len(alignment)
    seq_length = len(alignment[0].seq)
    site_taxa_ratio = seq_length / num_sequences

    invariant_count = 0

    for column in range(num_sequences):
        unique_characters = set()
        for record in alignment:
            try:
                unique_characters.add(record.seq[column])
            except IndexError:
                print("Index error occured, skipped " + msa_filepath)
        if len(unique_characters) == 1:
            invariant_count += 1

    percentage_invariant_sites = (invariant_count / num_sequences)

    return num_sequences, seq_length, site_taxa_ratio, percentage_invariant_sites


def gap_statistics(msa_filepath) -> (float, float):
    """
    Computes gap statistics for a reference MSA.

            Parameters:
                    :param msa_filepath: path to reference MSA

            Returns:
                    :return tuple: average gaps per sequence, standard deviation of gap count
    """
    alignment = AlignIO.read(msa_filepath, 'fasta')
    gap_counts = [seqe.seq.count('-') for seqe in alignment]
    seq_length = len(alignment[0].seq)

    mean_gaps = statistics.mean(gap_counts)
    avg_gaps = statistics.mean(gap_counts) / seq_length
    if mean_gaps != 0:
        cv_gaps = statistics.stdev(gap_counts) / mean_gaps
        if max(gap_counts) == min(gap_counts):
            normalized_gaps_counts = gap_counts
        else:
            normalized_gaps_counts = [(x - min(gap_counts)) / (max(gap_counts) - min(gap_counts)) for x in gap_counts]
        sk_gaps = skew(normalized_gaps_counts)
    else:
        cv_gaps = 0
        sk_gaps = 0
    kur_gaps = kurtosis(gap_counts, fisher=True)

    return avg_gaps, cv_gaps, kur_gaps, sk_gaps


def compute_entropy(msa_filepath, isAA):
    """
    Computes the per site entropies of the MSA as well as the mean of fractions for each nucleotide in the sequuences.

            Parameters:
                    :param msa_filepath: path to reference MSA

            Returns:
                    :return tuple: avg_entropy, std_entropy, min_entropy, max_entropy, g_fraction, c_fraction, a_fraction, t_fraction, rest_fraction
    """
    alignment = AlignIO.read(msa_filepath, 'fasta')
    site_entropies = []

    num_sites = alignment.get_alignment_length()

    nucleotides = ['A', 'C', 'T', 'G', '-']
    amino_acids = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

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

    summary_align = AlignInfo.SummaryInfo(alignment)
    consensus = summary_align.gap_consensus(threshold=0.7, ambiguous='-')
    byte_encoding = ''.join(format(ord(i), 'b').zfill(8) for i in consensus)

    approxEntropy = ApproximateEntropy.approximate_entropy_test(byte_encoding)
    approxEntropy_ape = approxEntropy[2]

    cumSum = CumulativeSums.cumulative_sums_test(byte_encoding)
    cumSum_p = cumSum[0]
    cumSum_abs_max = cumSum[2]
    cumSum_mode = cumSum[3]

    spec = SpectralTest.spectral_test(byte_encoding)
    spec_p = spec[0]
    spec_n1 = spec[2]
    spec_d = spec[3]

    matrix = Matrix.binary_matrix_rank_text(byte_encoding)
    matrix_p = matrix[0]

    complex_ = ComplexityTest.linear_complexity_test(byte_encoding)
    complex_p = complex_[0]
    complex_xObs = complex_[2]

    randex = RandomExcursions.random_excursions_test(byte_encoding)
    randex = [entry[3] for entry in randex]

    run = RunTest.run_test(byte_encoding)
    run_pi = run[2]
    run_vObs = run[3]

    run_one = RunTest.longest_one_block_test(byte_encoding)
    run_one_p = run_one[0]
    run_one_x0bs = run_one[2]
    run_one_mean = run_one[3]
    run_one_std = run_one[4]
    run_one_min = run_one[5]
    run_one_max = run_one[6]

    nucleotide_statistics = []
    if not isAA:
        for seq_record in alignment:
            nucleotide_counts = {nucleotide: seq_record.seq.upper().count(nucleotide) for nucleotide in nucleotides}

            if len(seq_record) != 0:
                g_fraction = nucleotide_counts["G"] / len(seq_record)
                c_fraction = nucleotide_counts["C"] / len(seq_record)
                a_fraction = nucleotide_counts["A"] / len(seq_record)
                t_fraction = nucleotide_counts["T"] / len(seq_record)
                rest_fraction = 1 - (g_fraction + c_fraction + a_fraction + t_fraction)

                nucleotide_statistics.append((g_fraction, c_fraction, a_fraction, t_fraction, rest_fraction))
        transposed = list(zip(*nucleotide_statistics))
        mean_values = [statistics.mean(values) for values in transposed]
        stats_aa = [0,0,0,0]
    else:
        amino_acid_counts = {aa: 0 for aa in amino_acids}
        amino_acid_statistics = []

        for seq_record in alignment:
            sequence = seq_record.seq.upper()

            for aa in amino_acids:
                aa_count = sequence.count(aa)
                amino_acid_counts[aa] += aa_count

            total_aa_count = sum(amino_acid_counts.values())
            aa_fractions = {aa: count / total_aa_count for aa, count in amino_acid_counts.items()}
            amino_acid_statistics.append(aa_fractions)

        mean_values = {aa: statistics.mean([stats[aa] for stats in amino_acid_statistics]) for aa in amino_acids}
        mean_values_list = list(mean_values.values())
        min_value = min(mean_values_list)
        max_value = max(mean_values_list)
        std_deviation = statistics.stdev(mean_values_list)
        mean_value = statistics.mean(mean_values_list)
        stats_aa = [min_value, max_value, std_deviation, mean_value]
        mean_values = [0,0,0,0,0]

    if isAA:
        for site in range(num_sites):
            site_column = alignment[:, site]

            if site_column == len(site_column) * 'N':  # if only N column
                continue
            if site_column == len(site_column) * '-':  # if only - column
                continue

            nucleotide_counts = {nucleotide: site_column.count(nucleotide) for nucleotide in nucleotides}
            total_count = sum(nucleotide_counts.values())

            if total_count != 0:
                probabilities = {}
                for nucleotide in nucleotides:
                    count = nucleotide_counts[nucleotide]
                    probabilities[nucleotide] = count / total_count

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

                entropy = entropy / math.log2(len(site_column))
                site_entropies.append(entropy)
    else:
        for site in range(num_sites):
            site_column = alignment[:, site]

            if site_column == len(site_column) * 'N':  # if only N column
                continue
            if site_column == len(site_column) * '-':  # if only - column
                continue

            aa_counts = {aa: site_column.count(aa) for aa in amino_acids}
            total_count = sum(aa_counts.values())

            if total_count != 0:
                probabilities = {}
                for aa in amino_acids:
                    count = aa_counts[aa]
                    probabilities[aa] = count / total_count

                entropy = 0.0
                for probability in probabilities.values():
                    if probability != 0:
                        entropy -= probability * math.log(probability, 2)
                try:
                    entropy = entropy / math.log2(len(site_column))
                except ZeroDivisionError:
                    entropy = 0
                site_entropies.append(entropy)

    max_entropy_ = np.max(site_entropies)
    avg_entropy_ = np.mean(site_entropies)
    std_entropy_ = np.std(site_entropies)
    skw_entropy_ = skew(site_entropies)
    kurtosis_entropy_ = kurtosis(site_entropies, fisher=True)

    return avg_entropy_, std_entropy_, max_entropy_, skw_entropy_, kurtosis_entropy_, mean_values[0], mean_values[1], \
           mean_values[2], \
           mean_values[3], mean_values[
               4], stats_aa[0],  stats_aa[1], stats_aa[2], stats_aa[3], approxEntropy_ape, cumSum_p, cumSum_abs_max, cumSum_mode, spec_p, spec_n1, spec_d, matrix_p, complex_p, \
           complex_xObs, randex[0], randex[1], randex[2], randex[3], randex[4], randex[5], randex[6], randex[
               7], run_pi, run_vObs, run_one_p, run_one_x0bs, run_one_mean, \
           run_one_std, run_one_min, run_one_max


if __name__ == '__main__':

    module_path = os.path.join(os.pardir, "configs/feature_config.py")

    feature_config = types.ModuleType('feature_config')
    feature_config.__file__ = module_path

    with open(module_path, 'rb') as module_file:
        code = compile(module_file.read(), module_path, 'exec')
        exec(code, feature_config.__dict__)

    loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
    filenames = loo_selection['verbose_name'].str.replace(".phy", "_reference.fasta").tolist()

    for file in filenames:
        if not os.path.exists(os.path.join(os.pardir, "data/raw/msa", file)):
            print("Not found: " + file)
            filenames.remove(file)

    if feature_config.INCUDE_TARA_BV_NEO:
        filenames = filenames + ["bv_reference.fasta", "neotrop_reference.fasta", "tara_reference.fasta"]

    results = []

    for file in filenames:
        filepath = os.path.join(os.pardir, "data/raw/msa", file)
        isAA = False
        datatype = loo_selection[loo_selection["verbose_name"] == file.replace("_reference.fasta", ".phy")].iloc[0][
            "data_type"]
        if datatype == "AA" or datatype == "DataType.AA":
            isAA = True
        print(isAA)  # Skip already processed
        print(file)
        avg_gaps, cv_gaps, kur_gaps, sk_gaps = gap_statistics(filepath)

        print(filepath)
        avg_entropy, std_entropy, max_entropy, skw_entropy, kurtosis_entropy, g_fraction, c_fraction, a_fraction, t_fraction, rest_fraction, aa_stat_min,  aa_stat_max, aa_stat_std, aa_stat_mean ,approxEntropy_ape_msa, cumSum_p_msa, cumSum_abs_max_msa, cumSum_mode_msa, spec_p_msa, spec_n1_msa, spec_d_msa, matrix_p_msa, complex_p_msa, \
        complex_xObs_msa, randex_0_msa, randex_1_msa, randex_2_msa, randex_3_msa, randex_4_msa, randex_5_msa, randex_6_msa, randex_7_msa, run_pi, run_vObs, run_one_p, run_one_x0bs, run_one_mean, \
        run_one_std_msa, run_one_min_msa, run_one_max_msa = compute_entropy(
            filepath, isAA)
        num_seq, seq_length, site_taxa_ratio, percentage_invariant_sites = basic_msa_features(filepath)

        name = ""

        if file == "neotrop_reference.fasta":
            name = "neotrop"
        elif file == "bv_reference.fasta":
            name = "bv"
        elif file == "tara_reference.fasta":
            name = "tara"
        else:
            name = file.replace("_reference.fasta", "")

        results.append(
            (name, avg_gaps, cv_gaps, kur_gaps, sk_gaps, avg_entropy, std_entropy, max_entropy, skw_entropy,
             kurtosis_entropy, num_seq,
             seq_length, g_fraction, c_fraction,
             a_fraction, t_fraction, rest_fraction, aa_stat_min,  aa_stat_max, aa_stat_std, aa_stat_mean,approxEntropy_ape_msa, cumSum_p_msa, cumSum_abs_max_msa,
             cumSum_mode_msa, spec_p_msa, spec_n1_msa, spec_d_msa, matrix_p_msa, complex_p_msa, complex_xObs_msa,
             randex_0_msa, randex_1_msa, randex_2_msa, randex_3_msa, randex_4_msa, randex_5_msa,
             randex_6_msa, randex_7_msa, run_pi, run_vObs, run_one_p, run_one_x0bs, run_one_mean, run_one_std_msa,
             run_one_min_msa, run_one_max_msa, site_taxa_ratio, percentage_invariant_sites))
    df = pd.DataFrame(results,
                      columns=["dataset", "avg_gaps_msa", "cv_gaps_msa", "kur_gaps_msa", "sk_gaps_msa",
                               "avg_entropy_msa", "std_entropy_msa",
                               "max_entropy_msa", "sk_entropy_msa", "kur_entropy_msa", "num_seq", "seq_length",
                               "g_fraction_msa", "c_fraction_msa",
                               "a_fraction_msa", "t_fraction_msa", "rest_fraction_msa", "aa_stat_min",  "aa_stat_max", "aa_stat_std", "aa_stat_mean","approxEntropy_ape_msa",
                               "cumSum_p_msa", "cumSum_abs_max_msa", "cumSum_mode_msa", "spec_p_msa",
                               "spec_n1_msa", "spec_d_msa", "matrix_p_msa", "complex_p_msa", "complex_xObs_msa",
                               "randex_0_msa", "randex_1_msa", "randex_2_msa",
                               "randex_3_msa", "randex_4_msa", "randex_5_msa", "randex_6_msa", "randex_7_msa",
                               "run_pi_msa", "run_vObs_msa", "run_one_p_msa", "run_one_x0bs_msa", "run_one_mean_msa",
                               "run_one_std_msa", "run_one_min_msa", "run_one_max_msa", "site_taxa_ratio_msa",
                               "percentage_invariant_sites_msa"])
    df.to_csv(os.path.join(os.pardir, "data/processed/features", "msa_features.csv"), index=False)
