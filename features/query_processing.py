import multiprocessing
import types
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
    isAA = False
    loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
    datatype = loo_selection[loo_selection["verbose_name"] == query_filepath.replace("_query.fasta", ".phy")].iloc[0][
        "data_type"]
    if datatype == "AA" or datatype == "DataType.AA":
        isAA = True
        print("Found AA")
    print(isAA)  # Skip already processed
    print(query_filepath)
    for record in alignment:

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

        results.append((name, record.id, gap_fraction, longest_gap_rel,
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
                                        "rest_fraction_query", "aa_stat_min_query","aa_stat_max_query","aa_stat_std_query", "aa_stat_mean_query","min_gap_query", "max_gap_query", "mean_gap_query",
                                        "cv_gap_query", "sk_gap_query", "kur_gap_query"])
    df.to_csv(os.path.join(os.pardir, "data/processed/features", "query_features.csv"))
