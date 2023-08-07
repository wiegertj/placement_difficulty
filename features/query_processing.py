import multiprocessing
import types
import pandas as pd
import os
from Bio import AlignIO

from uncertainty.ApproximateEntropy import ApproximateEntropy
from uncertainty.Complexity import ComplexityTest
from uncertainty.CumulativeSum import CumulativeSums
from uncertainty.FrequencyTest import FrequencyTest
from uncertainty.Matrix import Matrix
from uncertainty.RandomExcursions import RandomExcursions
from uncertainty.RunTest import RunTest
from uncertainty.Serial import Serial
from uncertainty.Spectral import SpectralTest



def gap_statistics(query_filepath) -> list:
    """
    Computes gap statistics for each query in the query file.

            Parameters:
                    :param query_filepath: path to query file

            Returns:
                    :return list: dataset, sampleId, gap_fraction, longest_gap_rel, average_gap_length
    """
    results = []
    filepath = os.path.join(os.pardir, "data/raw/query", query_filepath)
    alignment = AlignIO.read(filepath, 'fasta')
    for record in alignment:

        sequence = str(record.seq)
        seq_length = len(sequence)

        # gap fraction
        gap_count = sequence.count('-')
        gap_fraction = gap_count / seq_length

        # longest gap
        longest_gap = 0
        current_gap = 0

        for base in sequence:
            if base == '-':
                current_gap += 1
                longest_gap = max(longest_gap, current_gap)
            else:
                current_gap = 0

        longest_gap_rel = longest_gap / len(sequence)

        # average gap length
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

        # check if the last character is a gap
        if in_gap:
            total_gap_length += current_gap_length
            gap_count += 1
        if gap_count > 0:
            average_gap_length = total_gap_length / gap_count
        else:
            average_gap_length = 0
        byte_encoding = ''.join(format(ord(i),'b').zfill(8) for i in sequence)
        approxEntropy = ApproximateEntropy.approximate_entropy_test(byte_encoding)
        if approxEntropy[1] == True:
            approxEntropy = 1
        else:
            approxEntropy = 0
        cumSum = CumulativeSums.cumulative_sums_test(byte_encoding)
        if cumSum[1] == True:
            cumSum = 1
        else:
            cumSum = 0
        monBit = FrequencyTest.monobit_test(byte_encoding)
        if monBit[1] == True:
            monBit = 1
        else:
            monBit = 0
        spec = SpectralTest.spectral_test(byte_encoding)
        if spec[1] == True:
            spec = 1
        else:
            spec = 0
        serial = Serial.serial_test(byte_encoding)
        if serial[0][1] == True or serial[1][1] == True:
            serial = 1
        else:
            serial = 0
        matrix = Matrix.binary_matrix_rank_text(byte_encoding)
        if matrix[1] == True:
            matrix = 1
        else:
            matrix = 0
        complex_ = ComplexityTest.linear_complexity_test(byte_encoding)
        if complex_[1] == True:
            complex_ = 1
        else:
            complex_ = 0

        randex = RandomExcursions.random_excursions_test(byte_encoding)
        randex = [1 if entry[-1] else 0 for entry in randex]

        run = RunTest.run_test(byte_encoding)
        if run[1] == True:
            run = 1
        else:
            run = 0

        run_one = RunTest.longest_one_block_test(byte_encoding)
        if run_one[1] == True:
            run_one = 1
        else:
            run_one = 0



        name = ""

        if query_filepath == "neotrop_query_10k.fasta":
            name = "neotrop"
        elif query_filepath == "bv_query.fasta":
            name = "bv"
        elif query_filepath == "tara_query.fasta":
            name = "tara"
        else:
            name = query_filepath.replace("_query.fasta", "")

        results.append((name, record.id, gap_fraction, longest_gap_rel, average_gap_length / len(sequence), approxEntropy, cumSum, monBit, spec, serial, matrix, complex_, run, run_one,
                        randex[0], randex[1], randex[2], randex[3], randex[4], randex[5], randex[6], randex[7]))


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

    # Use the pool to parallelize the execution of gap_statistics function
    results = []
    counter = 0
    for result in pool.imap(gap_statistics, filenames):
        results.append(result)
        print(result)
        print(counter)
        counter += 1

    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()

    # Flatten the list of results if needed
    results = [item for sublist in results for item in sublist]

    df = pd.DataFrame(results, columns=["dataset", "sampleId", "gap_fraction", "longest_gap_rel", "average_gap_length", "approxEntropy", "cumSum", "monBit", "spec", "serial", "matrix", "complex", "run", "run_one"
                                        "randex-4", "randex-3", "randex-2", "randex-1", "randex1", "randex2", "randex3", "randex4"])
    df.to_csv(os.path.join(os.pardir, "data/processed/features", "query_features.csv"), index=False)
