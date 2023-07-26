import types
import pandas as pd
import os
from Bio import AlignIO


def gap_statistics(query_filepath) -> list:
    """
    Computes gap statistics for each query in the query file.

            Parameters:
                    :param query_filepath: path to query file

            Returns:
                    :return list: dataset, sampleId, gap_fraction, longest_gap_rel, average_gap_length
    """
    results = []
    filepath = os.path.join(os.pardir, "data/raw/query", file)
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

        name = ""

        if query_filepath == "neotrop_query_10k.fasta":
            name = "neotrop"
        elif query_filepath == "bv_query.fasta":
            name = "bv"
        elif query_filepath == "tara_query.fasta":
            name = "tara"
        else:
            name = query_filepath.replace("_query.fasta", "")

        results.append((name, record.id, gap_fraction, longest_gap_rel, average_gap_length / len(sequence)))

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

    if feature_config.INCUDE_TARA_BV_NEO:
        filenames = filenames + ["bv_reference.fasta", "neotrop_reference.fasta", "tara_reference.fasta"]

    results = []
    for file in filenames:
        results_file = gap_statistics(file)
        results.extend(results_file)
    df = pd.DataFrame(results, columns=["dataset", "sampleId", "gap_fraction", "longest_gap_rel", "average_gap_length"])
    df.to_csv(os.path.join(os.pardir, "data/processed/features", "query_features.csv"), index=False)
