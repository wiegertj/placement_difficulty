import math
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
    return num_sequences, seq_length


def gap_statistics(msa_filepath) -> (float, float):
    """
    Computes gap statistics for a reference MSA.

            Parameters:
                    :param msa_filepath: path to reference MSA

            Returns:
                    :return tuple: average gaps per sequence, standard deviation of gap count
    """
    alignment = AlignIO.read(msa_filepath, 'fasta')
    gap_counts = [seq.count('-') for seq in alignment]
    seq_length = len(alignment[0].seq)
    avg_gaps = statistics.mean(gap_counts) / seq_length
    std_gaps = statistics.stdev(gap_counts)

    return avg_gaps, std_gaps


def compute_entropy(msa_filepath):
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

    # Get randomness
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

    # Count the occurrences of each nucleotide
    nucleotide_statistics = []
    for seq_record in alignment:
        nucleotide_counts = {nucleotide: seq_record.upper().count(nucleotide) for nucleotide in nucleotides}

        if len(seq_record) != 0:
            g_fraction = nucleotide_counts["G"] / len(seq_record)
            c_fraction = nucleotide_counts["C"] / len(seq_record)
            a_fraction = nucleotide_counts["A"] / len(seq_record)
            t_fraction = nucleotide_counts["T"] / len(seq_record)
            rest_fraction = 1 - (g_fraction + c_fraction + a_fraction + t_fraction)

            nucleotide_statistics.append((g_fraction, c_fraction, a_fraction, t_fraction, rest_fraction))
    transposed = list(zip(*nucleotide_statistics))
    mean_values = [statistics.mean(values) for values in transposed]

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

    max_entropy_ = np.max(site_entropies)
    avg_entropy_ = np.mean(site_entropies)
    std_entropy_ = np.std(site_entropies)
    skw_entropy_ = skew(site_entropies)
    kurtosis_entropy_ = kurtosis(site_entropies, fisher=False)

    return avg_entropy_, std_entropy_, max_entropy_, skw_entropy_, kurtosis_entropy_, mean_values[0], mean_values[1], \
           mean_values[2], \
           mean_values[3], mean_values[
               4], approxEntropy_ape, cumSum_p, cumSum_abs_max, cumSum_mode, spec_p, spec_n1, spec_d, matrix_p, complex_p, \
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
    filenames = filenames[:2]

    if feature_config.INCUDE_TARA_BV_NEO:
        filenames = filenames + ["bv_reference.fasta", "neotrop_reference.fasta", "tara_reference.fasta"]

    results = []
    for file in filenames:
        filepath = os.path.join(os.pardir, "data/raw/msa", file)
        avg_gaps, std_gaps = gap_statistics(filepath)
        print(filepath)
        avg_entropy, std_entropy, max_entropy, skw_entropy, kurtosis_entropy, g_fraction, c_fraction, a_fraction, t_fraction, rest_fraction, approxEntropy_ape_msa, cumSum_p_msa, cumSum_abs_max_msa, cumSum_mode_msa, spec_p_msa, spec_n1_msa, spec_d_msa, matrix_p_msa, complex_p_msa, \
        complex_xObs_msa, randex_0_msa, randex_1_msa, randex_2_msa, randex_3_msa, randex_4_msa, randex_5_msa, randex_6_msa, randex_7_msa, run_pi, run_vObs, run_one_p, run_one_x0bs, run_one_mean, \
        run_one_std_msa, run_one_min_msa, run_one_max_msa = compute_entropy(
            filepath)
        num_seq, seq_length = basic_msa_features(filepath)

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
            (name, avg_gaps, std_gaps, avg_entropy, std_entropy, max_entropy, skw_entropy, kurtosis_entropy, num_seq,
             seq_length, g_fraction, c_fraction,
             a_fraction, t_fraction, rest_fraction, approxEntropy_ape_msa, cumSum_p_msa, cumSum_abs_max_msa,
             cumSum_mode_msa, spec_p_msa, spec_n1_msa, spec_d_msa, matrix_p_msa, complex_p_msa, complex_xObs_msa,
             randex_0_msa, randex_1_msa, randex_2_msa, randex_3_msa, randex_4_msa, randex_5_msa,
             randex_6_msa, randex_7_msa, run_pi, run_vObs, run_one_p, run_one_x0bs, run_one_mean, run_one_std_msa,
             run_one_min_msa, run_one_max_msa))
    df = pd.DataFrame(results, columns=["dataset", "avg_gaps", "std_gaps", "avg_entropy", "std_entropy",
                                        "max_entropy", "skw_entropy", "kurtosis_entropy", "num_seq", "seq_length",
                                        "g_fraction", "c_fraction",
                                        "a_fraction", "t_fraction", "rest_fraction", "approxEntropy_ape_msa",
                                        "cumSum_p_msa", "cumSum_abs_max_msa", "cumSum_mode_msa", "spec_p_msa",
                                        "spec_n1_msa", "spec_d_msa", "matrix_p_msa", "complex_p_msa", "complex_xObs_msa", "randex_0_msa", "randex_1_msa", "randex_2_msa",
                                        "randex_3_msa", "randex_4_msa", "randex_5_msa", "randex_6_msa", "randex_7_msa",
                                        "run_pi", "run_vObs", "run_one_p", "run_one_x0bs", "run_one_mean",
                                        "run_one_std_msa", "run_one_min_msa", "run_one_max_msa"])
    df.to_csv(os.path.join(os.pardir, "data/processed/features", "msa_features.csv"), index=False)
