import pandas as pd
import os
from Bio import AlignIO


def gap_statistics(query_filepath, max_gap_fraction):
    print(query_filepath)
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

        # AVG Gap length
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

        # Check if the last character is a gap
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

        results.append((name, record.id, gap_fraction, longest_gap_rel, average_gap_length))

    return results


if __name__ == '__main__':
    filepaths = ["neotrop_query_10k.fasta", "bv_query.fasta", "tara_query.fasta"]
    results = []
    for file in filepaths:
        results_file = gap_statistics(file, 0.3)
        # print(results_file)
        results.extend(results_file)
    print(len(results))
    print(len(results[0]))
    df = pd.DataFrame(results, columns=["dataset", "sampleId", "gap_fraction", "longest_gap_rel", "average_gap_length"])
    df.to_csv(os.path.join(os.pardir, "data/processed/features", "query_features.csv"), index=False)
