import os
import time
import types
import pandas as pd
import statistics
import multiprocessing
from itertools import product

import probables
from Bio import SeqIO
from probables import BloomFilter
from scipy.stats import skew, kurtosis

module_path = os.path.join(os.pardir, "configs/feature_config.py")
feature_config = types.ModuleType('feature_config')
feature_config.__file__ = module_path

with open(module_path, 'rb') as module_file:
    code = compile(module_file.read(), module_path, 'exec')
    exec(code, feature_config.__dict__)


def filter_gapped_kmers(sequence, isAA, k=feature_config.K_MER_LENGTH,
                        max_gap_percent=feature_config.K_MER_MAX_GAP_PERCENTAGE,
                        max_n_percentage=feature_config.K_MER_MAX_N_PERCENTAGE):
    """
    Returns a list of k-mers for the given sequence considering a max_gap_percentage.
    Ambiguity code gets resolved on the fly by considering each possible k-mer.

            Parameters:

                    :param sequence:  DNA sequence
                    :param k: k-mer length
                    :param max_gap_percent: maximum percentage of gaps in a k-mer to be valid
                    :param max_n_percentage: maximum percentage of 'N' nucleotides

            Returns:
                    :return kmers: list of k-mers

    """

    kmer_list = []
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

    if not isAA:

        for i in range(len(sequence) - int(k) + 1):

            kmer = sequence[i:i + int(k)]
            gap_count = kmer.count('-')
            if (kmer != ('N' * int(k))) and (kmer.count('N') / int(k) <= max_n_percentage):
                if gap_count / int(k) <= max_gap_percent:

                    ambiguous_positions = [i for i, char in enumerate(kmer) if char in nucleotide_ambiguity_code]

                    expanded_kmers = []
                    if ambiguous_positions:
                        combinations = product(
                            *(nucleotide_ambiguity_code[char] for char in kmer if char in nucleotide_ambiguity_code))
                        for combination in combinations:
                            expanded_kmer = list(kmer)
                            for position, nucleotide in zip(ambiguous_positions, combination):
                                expanded_kmer[position] = nucleotide
                            expanded_kmers.append(''.join(expanded_kmer))
                        kmer_list.extend(expanded_kmers)
                    else:
                        kmer_list.append(kmer)
    else:
        for i in range(len(sequence) - int(k) + 1):
            kmer = sequence[i:i + int(k)]
            gap_count = kmer.count('-')
            if gap_count / k <= max_gap_percent:
                kmer_list.append(kmer)

    return kmer_list


def compute_string_kernel_statistics(query, k=feature_config.K_MER_LENGTH,
                                     max_gap_percent=feature_config.K_MER_MAX_GAP_PERCENTAGE):
    """
    Computes string kernel using a bloom filter of the query and all the bloom filters of the MSA sequences.
    Then summary statistics for a hash kernels are computed.
    Ambiguity code gets resolved on the fly by considering each possible k-mer.

            Parameters:
                    :param query: DNA sequence
                    :param k: k-mer length
                    :param max_gap_percent: maximum percentage of a k-mer to be valid

            Returns:
                     :return tuple: (dataset, sampleId, min_kernel, max_kernel, mean_kernel, std_kernel)
    """
    kmers_query = filter_gapped_kmers(str(query.seq), isAA, k, max_gap_percent)
    print("#"*50)
    print(query.id)
    print(kmers_query)
    print("#"*50)

    query_bf = bloom_filter(set(kmers_query), len(kmers_query), feature_config.BLOOM_FILTER_FP_RATE)

    result_string_kernels = []
    for bloom_filter_ref in bloom_filters_MSA:

        if not (bloom_filter_ref[0] == query.id):  # For leave one out, dont compare to sequence it self
            hash_kernel = 0
            for kmer in set(kmers_query):
                hash_kernel += bloom_filter_ref[1].check(kmer) * query_bf.check(kmer)

            result_string_kernels.append(
                hash_kernel / len(set(kmers_query)))  # normalize by the number of k-mers in query

    # Compute summary statistics over string kernels as features
    min_kernel = min(result_string_kernels)
    max_kernel = max(result_string_kernels)
    mean_kernel = sum(result_string_kernels) / len(result_string_kernels)
    std_kernel = statistics.stdev(result_string_kernels)
    kur_kernel = kurtosis(result_string_kernels, fisher=True)
    sk_kernel = skew(result_string_kernels)

    return msa_file.replace("_reference.fasta",
                            ""), query.id, min_kernel, max_kernel, mean_kernel, std_kernel, sk_kernel, kur_kernel


def bloom_filter(filtered_kmers, size, fp_rate):
    """
    Returns a bloomfilter with the k-mers

            Parameters:
                    :param filtered_kmers: list of k-mers to be put into a bloom filter
                    :param size: size of the bloom filter
                    :param fp_rate: false positive rate of the bloom filter

            Returns:
                    :return bf: bloom filter with k-mers
    """

    bf_ = BloomFilter(size, fp_rate)

    for kmer in filtered_kmers:
        bf_.add(kmer)
    return bf_


def combine_partial_files(dataset):
    csv_files = [filename for filename in os.listdir(os.path.join(os.pardir, "data/processed/features")) if
                 filename.startswith(dataset) and filename.endswith(".csv") and filename.__contains__("kmer")]
    print(csv_files)

    combined_df = pd.concat(
        [pd.read_csv(os.path.join(os.pardir, "data/processed/features", file)) for file in csv_files])
    combined_df.to_csv(os.path.join(os.pardir, "data/processed/features",
                                    dataset.replace("_reference.fasta", "") + "_" + str(
                                        str(len(combined_df))) + ".csv"), index=False)
    print("Done")


def initializer(bloom_filters_MSA_, msa_file_, isAA_):
    global bloom_filters_MSA
    global msa_file
    global isAA
    bloom_filters_MSA = bloom_filters_MSA_
    msa_file = msa_file_
    isAA = isAA_


def monitor_progress(results):
    completed_count = 0
    total_count = len(results)
    start_time = time.time()

    while completed_count < total_count:
        completed_count = sum(result.ready() for result in results)

        elapsed_time = time.time() - start_time

        average_time_per_task = elapsed_time / completed_count if completed_count > 0 else 0

        remaining_count = total_count - completed_count
        remaining_time = average_time_per_task * remaining_count

        elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        remaining_time_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))

        progress = f"Completed: {completed_count}/{total_count} tasks"
        time_estimate = f"Elapsed time: {elapsed_time_str}, Remaining time: {remaining_time_str}"
        print(progress)
        print(time_estimate)

        time.sleep(feature_config.KMER_PROCESSING_VERBOSE)


def multiprocess_string_kernel(query_filename, isAA, bloom_filters_MSA_, msa_file_, interval):
    data = []
    counter = 0
    current_loo_targets = pd.read_csv(os.path.join(os.pardir, "data/processed/target", "loo_result_entropy.csv"))
    sampledData = current_loo_targets[current_loo_targets["dataset"] == msa_file_.replace("_reference.fasta", "")]["sampleId"].values.tolist()
    for query_record in SeqIO.parse(os.path.join(os.pardir, "data/raw/query", query_filename), 'fasta'):
        counter += 1
        if (interval - feature_config.KMER_PROCESSING_STEPSIZE) < counter <= interval:
            if query_record.name in sampledData:
                data.append(query_record)
        if counter > interval:
            break

    pool = multiprocessing.Pool(processes=1,initializer=initializer, initargs=(bloom_filters_MSA_, msa_file_, isAA))
    results_async = [pool.apply_async(compute_string_kernel_statistics, (item,)) for item in data]
    #monitor_progress(results_async)
    try:
        output = [result_.get() for result_ in results_async]
    except probables.exceptions.InitializationError:
        print("No ouput")
        return 0

    pool.close()
    pool.join()

    return output


if __name__ == '__main__':

    # combine_partial_files("neotrop")
    # combine_partial_files("bv")
    # combine_partial_files("tara")

    module_path = os.path.join(os.pardir, "configs/feature_config.py")

    feature_config = types.ModuleType('feature_config')
    feature_config.__file__ = module_path

    with open(module_path, 'rb') as module_file:
        code = compile(module_file.read(), module_path, 'exec')
        exec(code, feature_config.__dict__)

    loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
    loo_selection = loo_selection.drop_duplicates(subset=["verbose_name"], keep="first")
    filenames = loo_selection['verbose_name'].str.replace(".phy", "_reference.fasta").tolist()
    try:
        filenames.remove("11226_0_reference.fasta")
        filenames.remove("15849_4_reference.fasta")
        filenames.remove("15849_1_reference.fasta")
    except ValueError:
        print("Skipped")

    if feature_config.INCUDE_TARA_BV_NEO:
        filenames = filenames + ["bv_reference.fasta", "neotrop_reference.fasta", "tara_reference.fasta"]

    if multiprocessing.current_process().name == 'MainProcess':
        multiprocessing.freeze_support()

    filenames = filenames[-135:]

    counter_msa = 0
    for msa_file in ["13674_0_reference.fasta"]:
        isAA = False
        datatype = loo_selection[loo_selection["verbose_name"] == msa_file.replace("_reference.fasta", ".phy")].iloc[0]["data_type"]
        if datatype == "AA" or datatype == "DataType.AA":
            isAA = True
        print(isAA)
        print(str(counter_msa) + "/" + str(len(filenames)))
        counter_msa += 1
        print("started: " + msa_file)
        if msa_file == "neotrop_reference.fasta":
            query_file = msa_file.replace("reference.fasta", "query_10k.fasta")
        else:
            query_file = msa_file.replace("reference.fasta", "query.fasta")

        # Skip already processed
        potential_path = os.path.join(os.pardir, "data/processed/features",
                                      msa_file.replace("_reference.fasta", "") + "_kmer" + str(
                                          feature_config.K_MER_LENGTH) + "_0" + str(
                                          feature_config.K_MER_MAX_GAP_PERCENTAGE).replace(
                                          "0.",
                                          "") + "_" + str(
                                          1000) + ".csv")
        if os.path.exists(potential_path):
            print("Skipped: " + msa_file + " already processed")
            #continue

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

        # Check if too large
        if len(next(SeqIO.parse(os.path.join(os.pardir, "data/raw/msa", msa_file), 'fasta').records).seq) > 10000:
           print("Skipped " + msa_file + " too large")
           continue
        num_sequences = sum(1 for _ in SeqIO.parse(os.path.join(os.pardir, "data/raw/msa", msa_file), 'fasta'))

        #if num_sequences > 150:
        #   print("Skipped " + msa_file + " too large")
        #  continue

        # Create bloom filters for each sequence in the MSA
        print(msa_file)
        for record in SeqIO.parse(os.path.join(os.pardir, "data/raw/msa", msa_file), 'fasta'):
            kmers = filter_gapped_kmers(str(record.seq), isAA, feature_config.K_MER_LENGTH,
                                        feature_config.K_MER_MAX_GAP_PERCENTAGE, isAA)
            kmers = set(kmers)
            if len(kmers) == 0:
                print("Skipped")
                continue
            bf = bloom_filter(kmers, len(kmers), feature_config.BLOOM_FILTER_FP_RATE)
            bloom_filters_MSA.append((record.id, bf))

        print("Created Bloom Filter for MSAs ... ")

        # Parallel code to compute and store blocks of defined stepsize query samples
        while True:
            interval_start += feature_config.KMER_PROCESSING_STEPSIZE
            result_tmp = multiprocess_string_kernel(query_file, isAA, bloom_filters_MSA, msa_file, interval_start)
            if result_tmp != 0:

                results.extend(result_tmp)
                df = pd.DataFrame(results,
                                  columns=['dataset', 'sampleId', 'min_kmer_sim', 'max_kmer_sim', 'mean_kmer_sim',
                                           'std_kmer_sim', 'sk_kmer_sim', 'kur_kmer_sim'])
                df.to_csv(os.path.join(os.pardir, "data/processed/features",
                                       msa_file.replace("_reference.fasta", "") + "_kmer" + str(
                                           feature_config.K_MER_LENGTH) + "_0" + str(
                                           feature_config.K_MER_MAX_GAP_PERCENTAGE).replace("0.",
                                                                                            "") + "_" + str(
                                           interval_start) + ".csv"), index=False)
                results = []

            if interval_start >= no_queries or interval_start >= bound:  # stop if we reached bound or no query samples left
                break
