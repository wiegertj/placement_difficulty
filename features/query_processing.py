import multiprocessing
import random
import types
from collections import Counter

import pandas as pd
import os
from Bio import AlignIO
from uncertainty.ApproximateEntropy import ApproximateEntropy
from uncertainty.Complexity import ComplexityTest
from uncertainty.CumulativeSum import CumulativeSums
from uncertainty.Matrix import Matrix
from uncertainty.RandomExcursions import RandomExcursions
from uncertainty.RunTest import RunTest
from uncertainty.Spectral import SpectralTest
from scipy.stats import skew, kurtosis
import statistics
import numpy as np
from scipy.stats import kurtosis, skew

def query_statistics(query_filepath) -> list:
    """
    Computes gap statistics, nucleotide fractions and randomness scores for each query in the query file.

            Parameters:
                    :param query_filepath: path to query file

            Returns:
                    :return list:
    """
    results = []
    filepath = os.path.join(os.pardir, "data/raw/query", query_filepath)
    alignment = AlignIO.read(filepath, 'fasta')

    analyzed_sites_9 = []
    analyzed_sites_8 = []
    analyzed_sites_7 = []
    analyzed_sites_95 = []
    analyzed_sites_3 = []
    analyzed_sites_1 = []
    analyzed_sites_6 = []
    analyzed_sites_5 = []
    analyzed_sites_4 = []
    analyzed_sites_2 = []



    # Iterate over each position in the alignment
    for position in range(len(alignment[0])):
        # Extract characters (residues) at the current position for all sequences
        residues_at_position = [str(record.seq[position]) for record in alignment]

        # Count the occurrences of each character
        char_counts = Counter(residues_at_position)

        # Calculate the most frequent character and its count
        most_common_char, most_common_count = char_counts.most_common(1)[0]

        # Calculate the total count of all characters excluding gaps and "N"s
        total_count = sum(count for char, count in char_counts.items())

        # Calculate the proportion of the most frequent character
        proportion_most_common = most_common_count / total_count if total_count > 0 else 0

        # Check if the proportion is below the threshold and the character is not a gap or "N"
        if proportion_most_common < 0.9 or most_common_char in ['-', 'N']:
            analyzed_sites_9.append((0, most_common_char))
        else:
            analyzed_sites_9.append((1, most_common_char))

        if proportion_most_common < 0.6 or most_common_char in ['-', 'N']:
            analyzed_sites_6.append((0, most_common_char))
        else:
            analyzed_sites_6.append((1, most_common_char))

        if proportion_most_common < 0.5 or most_common_char in ['-', 'N']:
            analyzed_sites_5.append((0, most_common_char))
        else:
            analyzed_sites_5.append((1, most_common_char))

        if proportion_most_common < 0.4 or most_common_char in ['-', 'N']:
            analyzed_sites_4.append((0, most_common_char))
        else:
            analyzed_sites_4.append((1, most_common_char))

        if proportion_most_common < 0.2 or most_common_char in ['-', 'N']:
            analyzed_sites_2.append((0, most_common_char))
        else:
            analyzed_sites_2.append((1, most_common_char))

        if proportion_most_common < 0.1 or most_common_char in ['-', 'N']:
            analyzed_sites_1.append((0, most_common_char))
        else:
            analyzed_sites_1.append((1, most_common_char))

        if proportion_most_common < 0.3 or most_common_char in ['-', 'N']:
            analyzed_sites_3.append((0, most_common_char))
        else:
            analyzed_sites_3.append((1, most_common_char))

        # Check if the proportion is below the threshold and the character is not a gap or "N"
        if proportion_most_common < 0.8 or most_common_char in ['-', 'N']:
            analyzed_sites_8.append((0, most_common_char))
        else:
            analyzed_sites_8.append((1, most_common_char))

        # Check if the proportion is below the threshold and the character is not a gap or "N"
        if proportion_most_common < 0.7 or most_common_char in ['-', 'N']:
            analyzed_sites_7.append((0, most_common_char))
        else:
            analyzed_sites_7.append((1, most_common_char))

        # Check if the proportion is below the threshold and the character is not a gap or "N"
        if proportion_most_common < 0.95 or most_common_char in ['-', 'N']:
            analyzed_sites_95.append((0, most_common_char))
        else:
            analyzed_sites_95.append((1, most_common_char))

    isAA = False
    loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
    datatype = loo_selection[loo_selection["verbose_name"] == query_filepath.replace("_query.fasta", ".phy")].iloc[0][
        "data_type"]
    if datatype == "AA" or datatype == "DataType.AA":
        isAA = True
        print("Found AA")
    print(isAA)  # Skip already processed
    for record in alignment:
        gap_matches = 0
        total_gap_count = 0
        for i, (flag, char) in enumerate(analyzed_sites_8):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 0 and char in ["-", "N"]:
                total_gap_count += 1
                if str(record.seq)[i] == char:
                    gap_matches += 1

        match_counter_9 = 0
        total_inv_sites_9 = 0
        for i, (flag, char) in enumerate(analyzed_sites_9):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 1:
                total_inv_sites_9 += 1
            if flag == 1 and str(record.seq)[i] == char and char not in ['-', 'N']:
                match_counter_9 += 1

        match_counter_6 = 0
        total_inv_sites_6 = 0
        for i, (flag, char) in enumerate(analyzed_sites_6):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 1:
                total_inv_sites_6 += 1
            if flag == 1 and str(record.seq)[i] == char and char not in ['-', 'N']:
                match_counter_6 += 1

        match_counter_5 = 0
        total_inv_sites_5 = 0
        for i, (flag, char) in enumerate(analyzed_sites_5):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 1:
                total_inv_sites_5 += 1
            if flag == 1 and str(record.seq)[i] == char and char not in ['-', 'N']:
                match_counter_5 += 1

        match_counter_4 = 0
        total_inv_sites_4 = 0
        for i, (flag, char) in enumerate(analyzed_sites_4):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 1:
                total_inv_sites_4 += 1
            if flag == 1 and str(record.seq)[i] == char and char not in ['-', 'N']:
                match_counter_4 += 1

        match_counter_2 = 0
        total_inv_sites_2 = 0
        for i, (flag, char) in enumerate(analyzed_sites_2):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 1:
                total_inv_sites_2 += 1
            if flag == 1 and str(record.seq)[i] == char and char not in ['-', 'N']:
                match_counter_2 += 1


        match_counter_8 = 0
        total_inv_sites_8 = 0
        for i, (flag, char) in enumerate(analyzed_sites_8):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 1:
                total_inv_sites_9 += 1
            if flag == 1 and str(record.seq)[i] == char and char not in ['-', 'N']:
                match_counter_8 += 1

        transition_count = 0
        transversion_count = 0
        mut_count = 0
        fraction_char_rests = []
        for i, (flag, char) in enumerate(analyzed_sites_8):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 1:
                total_inv_sites_8 += 1
            if flag == 1 and str(record.seq)[i] == char and char not in ['-', 'N']:
                match_counter_8 += 1
            if flag == 1 and str(record.seq)[i] != char:
                mut_count += 1
                if char in ["C", "T", "U"]:
                    if str(record.seq)[i] in ["A", "G"]:
                        transversion_count += 1
                elif char in ["A", "G"]:
                    if str(record.seq)[i] in ["C", "T", "U"]:
                        transversion_count += 1
                else:
                    transition_count += 1

                residues_at_position = [str(record.seq[i]) for record in alignment]
                residue_counts = Counter(residues_at_position)
                most_common_residue, most_common_count = residue_counts.most_common(1)[0]
                residues_at_position_del_most_common = [r for r in residues_at_position if r != most_common_residue]
                if str(record.seq)[i] in residues_at_position_del_most_common:
                    count_char = residues_at_position_del_most_common.count(str(record.seq)[i])
                    fraction_char_rest = count_char / len(residues_at_position_del_most_common)
                else:
                    fraction_char_rest = 0
                fraction_char_rests.append(fraction_char_rest)

        match_counter_7 = 0
        total_inv_sites_7 = 0
        for i, (flag, char) in enumerate(analyzed_sites_7):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 1:
                total_inv_sites_7 += 1
            if flag == 1 and str(record.seq)[i] == char and char not in ['-', 'N']:
                match_counter_7 += 1

        match_counter_95 = 0
        total_inv_sites_95 = 0
        for i, (flag, char) in enumerate(analyzed_sites_95):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 1:
                total_inv_sites_95 += 1
            if flag == 1 and str(record.seq)[i] == char and char not in ['-', 'N']:
                match_counter_95 += 1

        match_counter_1 = 0
        total_inv_sites_1 = 0
        for i, (flag, char) in enumerate(analyzed_sites_1):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 1:
                total_inv_sites_1 += 1
            if flag == 1 and str(record.seq)[i] == char and char not in ['-', 'N']:
                match_counter_1 += 1

        match_counter_3 = 0
        total_inv_sites_3 = 0
        for i, (flag, char) in enumerate(analyzed_sites_3):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 1:
                total_inv_sites_3 += 1
            if flag == 1 and str(record.seq)[i] == char and char not in ['-', 'N']:
                match_counter_3 += 1

        sequence = str(record.seq)
        seq_length = len(sequence)
        if not isAA:

            nucleotides = ['A', 'C', 'T', 'G', '-']

            nucleotide_counts = {nucleotide: sequence.upper().count(nucleotide) for nucleotide in nucleotides}

            g_fraction = nucleotide_counts["G"] / len(sequence)
            c_fraction = nucleotide_counts["C"] / len(sequence)
            a_fraction = nucleotide_counts["A"] / len(sequence)
            t_fraction = nucleotide_counts["T"] / len(sequence)
            rest_fraction = 1 - (g_fraction + c_fraction + a_fraction + t_fraction)
            aa_stats = [0, 0, 0, 0]
            mean_values = [g_fraction, c_fraction, a_fraction, t_fraction, rest_fraction]
        else:
            amino_acids = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V",
                           "W", "Y"]
            amino_acid_counts = {aa: 0 for aa in amino_acids}
            amino_acid_statistics = []

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
            aa_stats = [min_value, max_value, std_deviation, mean_value]
            mean_values = [0, 0, 0, 0, 0]

        gap_count = sequence.count('-')
        gap_fraction = gap_count / seq_length

        num_parts = 10

        part_length = len(sequence) // num_parts

        gap_fractions = []
        gap_positions = []

        for i in range(num_parts):
            start_pos = i * part_length
            end_pos = (i + 1) * part_length
            part_sequence = sequence[start_pos:end_pos]

            gap_count = part_sequence.count("-")
            gap_fraction_ = gap_count / part_length

            gap_fractions.append(gap_fraction_)
            gap_positions.append((start_pos, end_pos))

        longest_gap = 0
        current_gap = 0

        for base in sequence:
            if base == '-':
                current_gap += 1
                longest_gap = max(longest_gap, current_gap)
            else:
                current_gap = 0

        longest_gap_rel = longest_gap / len(sequence)

        in_gap = False
        gap_lengths = []
        current_gap_length = 0

        for char in sequence:
            if char == "-":
                if in_gap:
                    current_gap_length += 1
                else:
                    in_gap = True
                    current_gap_length = 1
            else:
                if in_gap:
                    gap_lengths.append(current_gap_length)
                    in_gap = False

        number_of_gaps = len(gap_lengths)
        if len(gap_lengths) >= 1:
            min_gap = min(gap_lengths) / len(sequence)
            max_gap = max(gap_lengths) / len(sequence)
            mean_gap = statistics.mean(gap_lengths)
            kur_gap = kurtosis(gap_lengths, fisher=True)
            if min(gap_lengths) != max(gap_lengths):
                sk_gap = skew([(x - min(gap_lengths)) / (max(gap_lengths) - min(gap_lengths)) for x in
                               gap_lengths])
            else:
                sk_gap = 0
        else:
            min_gap = 0
            max_gap = 0
            mean_gap = 0
            kur_gap = 0
            sk_gap = 0

        if mean_gap != 0 and len(gap_lengths) > 1:
            cv_gap = statistics.stdev(gap_lengths) / mean_gap
        else:
            cv_gap = 0

        byte_encoding = ''.join(format(ord(i), 'b').zfill(8) for i in sequence)

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

        name = ""

        if query_filepath == "neotrop_query_10k.fasta":
            name = "neotrop"
        elif query_filepath == "bv_query.fasta":
            name = "bv"
        elif query_filepath == "tara_query.fasta":
            name = "tara"
        else:
            name = query_filepath.replace("_query.fasta", "")

        if total_inv_sites_7 > 0:
            match_rel_7 = match_counter_7 / total_inv_sites_7
        else:
            match_rel_7 = 0

        if total_inv_sites_8 > 0:
            match_rel_8 = match_counter_8 / total_inv_sites_8
        else:
            match_rel_8 = 0

        if total_inv_sites_9 > 0:
            match_rel_9 = match_counter_9 / total_inv_sites_9
        else:
            match_rel_9 = 0

        if total_inv_sites_95 > 0:
            match_rel_95 = match_counter_95 / total_inv_sites_95
        else:
            match_rel_95 = 0

        if total_inv_sites_3 > 0:
            match_rel_3 = match_counter_3 / total_inv_sites_3
        else:
            match_rel_3 = 0

        if total_inv_sites_6 > 0:
            match_rel_6 = match_counter_6 / total_inv_sites_6
        else:
            match_rel_6 = 0

        if total_inv_sites_5 > 0:
            match_rel_5 = match_counter_5 / total_inv_sites_5
        else:
            match_rel_5 = 0

        if total_inv_sites_4 > 0:
            match_rel_4 = match_counter_4 / total_inv_sites_4
        else:
            match_rel_4 = 0

        if total_inv_sites_2 > 0:
            match_rel_2 = match_counter_2 / total_inv_sites_2
        else:
            match_rel_2 = 0

        if total_inv_sites_1 > 0:
            match_rel_1 = match_counter_1 / total_inv_sites_1
        else:
            match_rel_1 = 0

        if total_gap_count > 0:
            match_rel_gap = gap_matches / total_gap_count
        else:
            match_rel_gap = 0

        if mut_count > 0:
            transition_count_rel = transition_count / mut_count
            transversion_count_rel = transversion_count / mut_count
        else:
            transition_count_rel = 0
            transversion_count_rel = 0

        if len(fraction_char_rests) > 0:
            max_fraction_char_rests = np.max(fraction_char_rests)
            min_fraction_char_rests = np.min(fraction_char_rests)
            avg_fraction_char_rests = np.mean(fraction_char_rests)
            std_fraction_char_rests = np.std(fraction_char_rests)
            skw_fraction_char_rests = skew(fraction_char_rests)
            kur_fraction_char_rests = kurtosis(fraction_char_rests, fisher=True)
        else:
            max_fraction_char_rests = 0.0
            min_fraction_char_rests = 0.0
            avg_fraction_char_rests = 0.0
            std_fraction_char_rests = 0.0
            skw_fraction_char_rests = 0.0
            kur_fraction_char_rests = 0.0

        results.append((name, record.id, gap_fraction, longest_gap_rel,
                        match_counter_7 / seq_length, match_counter_8 / seq_length, match_counter_9 / seq_length,
                        match_counter_95 / seq_length, match_counter_3 / seq_length, match_counter_1 / seq_length,
                        match_rel_7, match_rel_8, match_rel_9, match_rel_95, match_rel_3, match_rel_1, match_rel_gap,
                        match_rel_2, match_rel_4, match_rel_6, match_rel_5,
                        transition_count_rel, transversion_count_rel, max_fraction_char_rests,
                        min_fraction_char_rests, avg_fraction_char_rests, std_fraction_char_rests,
                        skw_fraction_char_rests, kur_fraction_char_rests,
                        gap_fractions[0], gap_fractions[1], gap_fractions[2], gap_fractions[3], gap_fractions[4],
                        gap_fractions[5], gap_fractions[6],
                        gap_fractions[7], gap_fractions[8], gap_fractions[9],
                        approxEntropy_ape, cumSum_p, cumSum_abs_max, cumSum_mode, spec_p, spec_d, spec_n1, matrix_p,
                        complex_p, complex_xObs, run_pi, run_vObs,
                        run_one_p, run_one_x0bs, run_one_mean, run_one_std, run_one_min, run_one_max,
                        randex[0], randex[1], randex[2], randex[3], randex[4], randex[5], randex[6], randex[7],
                        mean_values[0], mean_values[1], mean_values[2], mean_values[3], mean_values[4],
                        aa_stats[0], aa_stats[1], aa_stats[2], aa_stats[3], min_gap, max_gap, mean_gap,
                        cv_gap, sk_gap, kur_gap))

    return results


if __name__ == '__main__':

    if multiprocessing.current_process().name == 'MainProcess':
        multiprocessing.freeze_support()

    module_path = os.path.join(os.pardir, "configs/feature_config.py")

    feature_config = types.ModuleType('feature_config')
    feature_config.__file__ = module_path

    with open(module_path, 'rb') as module_file:
        code = compile(module_file.read(), module_path, 'exec')
        exec(code, feature_config.__dict__)

    loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
    loo_selection = loo_selection.drop_duplicates(subset=['verbose_name'], keep='first')
    filenames = loo_selection['verbose_name'].str.replace(".phy", "_query.fasta").to_list()

    if feature_config.INCUDE_TARA_BV_NEO:
        filenames = filenames + ["bv_query.fasta", "neotrop_query_10k.fasta", "tara_query.fasta"]

    print(len(filenames))

    for file in filenames:
        if not os.path.exists(os.path.join(os.pardir, "data/raw/query", file)):
            print("Query file not found: " + file)
            filenames.remove(file)

    num_processes = multiprocessing.cpu_count()  # You can adjust the number of processes as needed
    pool = multiprocessing.Pool(processes=num_processes)

    results = []
    counter = 0

    for result in pool.imap(query_statistics, filenames):
        results.append(result)
        print(counter)
        counter += 1

    pool.close()
    pool.join()

    results = [item for sublist in results for item in sublist]

    df = pd.DataFrame(results, columns=["dataset", "sampleId", "gap_fraction", "longest_gap_rel",
                                        "frac_inv_sites_msa7", "frac_inv_sites_msa8", "frac_inv_sites_msa9",
                                        "frac_inv_sites_msa95", "frac_inv_sites_msa3", "frac_inv_sites_msa1",
                                        "match_rel_7", "match_rel_8", "match_rel_9", "match_rel_95", "match_rel_3",
                                        "match_rel_1", "match_rel_gap", "match_rel_2", "match_rel_4", "match_rel_6", "match_rel_5","transition_count_rel",
                                        "transversion_count_rel", "max_fraction_char_rests",
                                        "min_fraction_char_rests", "avg_fraction_char_rests", "std_fraction_char_rests",
                                        "skw_fraction_char_rests", "kur_fraction_char_rests",
                                        "gap_positions_0", "gap_positions_1", "gap_positions_2", "gap_positions_3",
                                        "gap_positions_4", "gap_positions_5", "gap_positions_6",
                                        "gap_positions_7", "gap_positions_8", "gap_positions_9",
                                        "approxEntropy_ape_query", "cumSum_p_query", "cumSum_abs_max_query",
                                        "cumSum_mode_query", "spec_p_query",
                                        "spec_d_query", "spec_n1_query", "matrix_p_query", "complex_p_query",
                                        "complex_xObs_query", "run_pi_query",
                                        "run_vObs_query",
                                        "run_one_p_query", "run_one_x0bs_query", "run_one_mean_query",
                                        "run_one_std_query", "run_one_min_query",
                                        "run_one_max_query",
                                        "randex-4_query", "randex-3_query", "randex-2_query", "randex-1_query",
                                        "randex1_query", "randex2_query",
                                        "randex3_query", "randex4_query", "g_fraction_query",
                                        "a_fraction_query", "t_fraction_query", "c_fraction_query",
                                        "rest_fraction_query", "aa_stat_min_query", "aa_stat_max_query",
                                        "aa_stat_std_query", "aa_stat_mean_query", "min_gap_query", "max_gap_query",
                                        "mean_gap_query",
                                        "cv_gap_query", "sk_gap_query", "kur_gap_query"])
    df.to_csv(os.path.join(os.pardir, "data/processed/features", "query_features.csv"))
