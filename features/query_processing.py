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
    for record in alignment:

        sequence = str(record.seq)
        seq_length = len(sequence)

        nucleotides = ['A', 'C', 'T', 'G', '-']
        nucleotide_counts = {nucleotide: sequence.upper().count(nucleotide) for nucleotide in nucleotides}

        g_fraction = nucleotide_counts["G"] / len(sequence)
        c_fraction = nucleotide_counts["C"] / len(sequence)
        a_fraction = nucleotide_counts["A"] / len(sequence)
        t_fraction = nucleotide_counts["T"] / len(sequence)
        rest_fraction = 1 - (g_fraction + c_fraction + a_fraction + t_fraction)

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
            gap_fraction = gap_count / part_length

            gap_fractions.append(gap_fraction)
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

        total_gap_length = 0
        gap_count = 0

        in_gap = False
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
                    total_gap_length += current_gap_length
                    gap_count += 1
                    in_gap = False

        if in_gap:
            total_gap_length += current_gap_length
            gap_count += 1
        if gap_count > 0:
            average_gap_length = total_gap_length / gap_count
        else:
            average_gap_length = 0
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

        results.append((name, record.id, gap_fraction, longest_gap_rel, average_gap_length / len(sequence),
                        gap_positions[0], gap_positions[1], gap_positions[2], gap_positions[3], gap_positions[4], gap_positions[5], gap_positions[6],
                        gap_positions[7], gap_positions[8], gap_positions[9],
                        approxEntropy_ape, cumSum_p, cumSum_abs_max, cumSum_mode, spec_p, spec_d, spec_n1, matrix_p,
                        complex_p, complex_xObs, run_pi, run_vObs,
                        run_one_p, run_one_x0bs, run_one_mean, run_one_std, run_one_min, run_one_max,
                        complex_,
                        randex[0], randex[1], randex[2], randex[3], randex[4], randex[5], randex[6], randex[7],
                        g_fraction, a_fraction, t_fraction, c_fraction, rest_fraction))

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
    filenames = loo_selection['verbose_name'].str.replace(".phy", "_query.fasta").tolist()

    if feature_config.INCUDE_TARA_BV_NEO:
        filenames = filenames + ["bv_query.fasta", "neotrop_query_10k.fasta", "tara_query.fasta"]

    print(len(filenames))
    filenames = filenames[:2]

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

    df = pd.DataFrame(results, columns=["dataset", "sampleId", "gap_fraction", "longest_gap_rel", "average_gap_length",
                                        "gap_positions_0", "gap_positions_1", "gap_positions_2", "gap_positions_3",
                                        "gap_positions_4", "gap_positions_5", "gap_positions_6",
                                        "gap_positions_7", "gap_positions_8", "gap_positions_9",
                                        "approxEntropy_ape", "cumSum_p", "cumSum_abs_max", "cumSum_mode", "spec_p",
                                        "spec_d", "spec_n1", "matrix_p", "complex_p", "complex_xObs", "run_pi",
                                        "run_vObs",
                                        "run_one_p", "run_one_x0bs", "run_one_mean", "run_one_std", "run_one_min",
                                        "run_one_max",
                                        "complex_",
                                        "randex-4", "randex-3", "randex-2", "randex-1", "randex1", "randex2",
                                        "randex3", "randex4", "g_fraction",
                                        "a_fraction", "t_fraction", "c_fraction", "rest_fraction"])
    df.to_csv(os.path.join(os.pardir, "data/processed/features", "query_features.csv"))
