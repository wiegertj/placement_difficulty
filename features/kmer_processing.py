import os
import time
import pandas as pd
import statistics
import multiprocessing
import configs.feature_config as config
from itertools import product
from Bio import SeqIO
from probables import BloomFilter


def filter_gapped_kmers(sequence, k=config.K_MER_LENGTH, max_gap_percent=config.K_MER_MAX_GAP_PERCENTAGE) -> list:
    """
    Returns a list of k-mers for the given sequence considering a max_gap_percentage.
    Ambiguity code gets resolved on the fly by considering each possible k-mer.

            Parameters:
                    sequence (string): DNA sequence
                    k (int): k-mer length
                    max_gap_percent (float): maximum percentage of gaps in a k-mer to be valid

            Returns:
                    kmers (list): list of k-mers
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
        'V': ['A', 'C', 'G']
    }

    for i in range(len(sequence) - k + 1):

        kmer = sequence[i:i + k]
        gap_count = kmer.count('-')
        if gap_count / k <= max_gap_percent:

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
    return kmer_list


def compute_string_kernel_statistics(query, k=config.K_MER_LENGTH, max_gap_percent=config.K_MER_MAX_GAP_PERCENTAGE) -> (str, str, float, float, float, float):
    """
    Computes string kernel using a bloom filter of the query and all the bloom filters of the MSA sequences.
    Then summary statistics for a hash kernels are computed.
    Ambiguity code gets resolved on the fly by considering each possible k-mer.

            Parameters:
                    query (string): DNA sequence
                    k (int): k-mer length
                    max_gap_percent (float): maximum percentage of a k-mer to be valid

            Returns:
                     tuple: (dataset, sampleId, min_kernel, max_kernel, mean_kernel, std_kernel)
    """
    kmers_query = filter_gapped_kmers(str(query.seq), k, max_gap_percent)
    query_bf = bloom_filter(kmers_query, len(kmers_query), config.BLOOM_FILTER_FP_RATE)

    result_string_kernels = []
    for bloom_filter_ref in bloom_filters_MSA:
        hash_kernel = 0
        for kmer in set(kmers_query):
            hash_kernel += bloom_filter_ref.check(kmer) * query_bf.check(kmer)

        result_string_kernels.append(hash_kernel / len(kmers_query))  # normalize by the number of k-mers in query

    # Compute summary statistics over string kernels as features
    min_kernel = min(result_string_kernels)
    max_kernel = max(result_string_kernels)
    mean_kernel = sum(result_string_kernels) / len(result_string_kernels)
    std_kernel = statistics.stdev(result_string_kernels)

    return msa_file.replace("_reference.fasta", ""), query.id, min_kernel, max_kernel, mean_kernel, std_kernel


def bloom_filter(filtered_kmers, size, fp_rate):
    """
    Returns a bloomfilter with the k-mers

            Parameters:
                    filtered_kmers (list): list of k-mers to be put into a bloom filter
                    size (int): size of the bloom filter
                    fp_rate (float): false positive rate of the bloom filter

            Returns:
                    bf (BloomFilter): bloom filter with k-mers
    """

    bf_ = BloomFilter(size, fp_rate)

    for kmer in filtered_kmers:
        bf_.add(kmer)
    return bf_


# ----------------------------------------------- HELPER -----------------------------------------------


def initializer(bloom_filters_MSA_, msa_file_):
    global bloom_filters_MSA
    global msa_file
    bloom_filters_MSA = bloom_filters_MSA_
    msa_file = msa_file_


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

        time.sleep(config.KMER_PROCESSING_VERBOSE)


def multiprocess_string_kernel(query_filename, bloom_filters_MSA_, msa_file_, interval):
    data = []
    counter = 0
    for query_record in SeqIO.parse(os.path.join(os.pardir, "data/raw/query", query_filename), 'fasta'):
        counter += 1
        if (interval - config.KMER_PROCESSING_STEPSIZE) < counter <= interval:
            data.append(query_record)
        if counter > interval:
            break

    pool = multiprocessing.Pool(initializer=initializer, initargs=(bloom_filters_MSA_, msa_file_))
    results_async = [pool.apply_async(compute_string_kernel_statistics, (item,)) for item in data]
    monitor_progress(results_async)
    output = [result_.get() for result_ in results_async]

    pool.close()
    pool.join()

    return output


# ----------------------------------------------- MAIN -----------------------------------------------


if __name__ == '__main__':

    interval_start = config.KMER_PROCESSING_INTERVAL_START  # sequence number to start with in query (last file number)
    bound = config.KMER_PROCESSING_COUNT  # how many sequences



    if multiprocessing.current_process().name == 'MainProcess':
        multiprocessing.freeze_support()

    for msa_file, query_file in [("neotrop_reference.fasta", "neotrop_query_10k.fasta")]:

        results = []

        bloom_filters_MSA = []
        string_kernel_features = []

        no_queries = len(list(SeqIO.parse(os.path.join(os.pardir, "data/raw/query", query_file), 'fasta').records))

        # Create bloom filters for each sequence in the MSA
        for record in SeqIO.parse(os.path.join(os.pardir, "data/raw/msa", msa_file), 'fasta'):
            kmers = filter_gapped_kmers(str(record.seq), config.K_MER_LENGTH, config.K_MER_MAX_GAP_PERCENTAGE)
            bf = bloom_filter(kmers, len(kmers), config.BLOOM_FILTER_FP_RATE)
            bloom_filters_MSA.append(bf)

        # Parallel code to compute and store blocks of defined stepsize query samples
        while True:
            interval_start += config.KMER_PROCESSING_STEPSIZE
            result_tmp = multiprocess_string_kernel(query_file, bloom_filters_MSA, msa_file, interval_start)
            results.extend(result_tmp)
            df = pd.DataFrame(results,
                              columns=['dataset', 'sampleId', 'min_kernel', 'max_kernel', 'mean_kernel', 'std_kernel'])
            df.to_csv(os.path.join(os.pardir, "data/processed/features",
                                   msa_file.replace("_reference.fasta", "") + "_kmer" + str(
                                       config.K_MER_LENGTH) + "_0" + str(config.K_MER_MAX_GAP_PERCENTAGE).replace("0.",
                                                                                                                  "") + "_" + str(
                                       interval_start) + ".csv"), index=False)
            results = []

            if interval_start >= no_queries or interval_start >= bound:  # stop if we reached bound or no query samples left
                break
