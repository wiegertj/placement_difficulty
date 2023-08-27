import types
import pandas as pd
import os
import statistics
from Bio import SeqIO


def compute_hamming_distance(msa_file, query_file) -> list:
    """
    Computes for each sequence in the query file all the hamming distances to the sequences in the MSA.
    Then for each query sequence summary statistics are computed.

            Parameters:
                    :param msa_file: path to reference MSA file
                    :param query_file: path to query file

            Returns:
                    :return list of: dataset, sampleId, relative min hamming dist, relative max hamming dist, relative average hamming distance, relative standard deviation hamming distance
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

    module_path = os.path.join(os.pardir, "configs/feature_config.py")

    feature_config = types.ModuleType('feature_config')
    feature_config.__file__ = module_path

    with open(module_path, 'rb') as module_file:
        code = compile(module_file.read(), module_path, 'exec')
        exec(code, feature_config.__dict__)

    loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
    filenames = loo_selection['verbose_name'].str.replace(".phy", "_reference.fasta").tolist()

    for file in filenames:
        if not os.path.exists(file.replace("reference.fasta", "query.fasta")):
            print("Not found: " + file)
            filenames.remove(file)

    if feature_config.INCUDE_TARA_BV_NEO:
        filenames = filenames + ["bv_reference.fasta", "neotrop_reference.fasta", "tara_reference.fasta"]

    counter_msa = 0
    for msa_file in filenames:
        print(msa_file)
        if os.path.isfile(os.path.join(os.pardir, "data/processed/features",
                                       msa_file.replace("reference.fasta", "msa_dist.csv"))):
            print("Found " + msa_file + " skipped")
            continue
        if len(next(SeqIO.parse(os.path.join(os.pardir, "data/raw/msa", msa_file), 'fasta').records).seq) > 10000:
            print("Skipped " + msa_file + " too large")
            continue

        print(str(counter_msa) + "/" + str(len(filenames)))
        counter_msa += 1
        result_tmp = compute_hamming_distance(msa_file, msa_file.replace("reference.fasta", "query.fasta"))

        df = pd.DataFrame(result_tmp, columns=['dataset', 'sampleId', 'min_ham_dist', 'max_ham_dist', 'avg_ham_dist',
                                               'std_ham_dist'])
        df.to_csv(os.path.join(os.pardir, "data/processed/features",
                               msa_file.replace("_reference.fasta", "") + "_msa_dist.csv"))
