#!/usr/bin/env python3
import collections
import os
import statistics
import sys
from scipy.stats import kurtosis, skew
import ete3
import pandas as pd
from io import StringIO

from Bio import Phylo, AlignIO, SeqIO

"""
Our cheap MSA parser stuff 
"""


class MSAParser:
    def __init__(self):
        pass

    def parse(self, msa_path):
        return []

    def parse_and_fix(self, msa_path, data_type):
        sequences = self.parse(msa_path)
        if data_type != "DNA":
            return sequences

        for sequence in sequences:
            sequence.sequence = sequence.sequence.upper()
            sequence.sequence = sequence.sequence.replace("N", "-")
            sequence.sequence = sequence.sequence.replace("?", "-")
            sequence.sequence = sequence.sequence.replace("X", "-")
            sequence.sequence = sequence.sequence.replace("U", "T")
        return sequences

    def __str__(self):
        return type(self).__name__


class Sequence:
    def __init__(self, id, sequence, comment=""):
        self.id = id
        self.sequence = sequence
        self.comment = comment


class PhylipsParser(MSAParser):
    def parse(self, msa_path):
        with open(msa_path) as file:
            first_line = True
            num_taxa = sequence_len = 0
            sequences = []

            for line in file:
                line = line.strip()
                temp_line = line.split()

                if first_line:
                    num_taxa = int(temp_line[0])
                    sequence_len = int(temp_line[1])
                    first_line = False
                    continue
                if not line:
                    continue

                sequence = temp_line[-1]
                taxon = " ".join(temp_line[:-1])

                if len(sequence) != sequence_len:
                    raise ValueError(f"unexpected sequence length: {len(sequence)} vs {sequence_len}")
                sequences.append(Sequence(taxon, sequence))
            if len(sequences) != num_taxa:
                raise ValueError(f"unexpected number of sequences: {len(sequences)} vs {num_taxa}")
        return sequences


class FastaParser(MSAParser):
    def parse(self, msa_path):
        with open(msa_path) as file:
            sequences = []
            current_taxon = ""
            current_sequence = ""
            current_comment = ""

            first_line = True

            for line in file:
                line = line.strip()
                if not line:
                    continue

                if first_line:
                    if not line.startswith(">"):
                        raise ValueError(f"taxon name is expected to start with '>'")
                    first_line = False

                if line.startswith(">"):
                    sequence = Sequence(current_taxon, current_sequence, comment=current_comment)
                    if current_sequence:
                        sequences.append(sequence)

                    temp_line = line.split()
                    current_taxon = "".join(list(temp_line[0])[1:])
                    current_comment = " ".join(temp_line[1:])
                    current_sequence = ""
                else:
                    current_sequence += line

            if current_sequence:
                sequence = Sequence(current_taxon, current_sequence, comment=current_comment)
                sequences.append(sequence)

            sequence_len = 0
            for sequence in sequences:
                if not sequence_len:
                    sequence_len = len(sequence.sequence)
                if sequence_len != len(sequence.sequence):
                    raise ValueError(f"unexpected sequence length: {len(sequence.sequence)} vs {sequence_len}")
        return sequences


class PhylipParser(MSAParser):
    def parse(self, msa_path):
        with open(msa_path) as file:
            first_line = True
            seq_names = []
            sequence_dict = collections.defaultdict(lambda: "")
            sequences = []
            num_taxa = seq_len = counter = 0

            for line in file:
                line = line.strip()
                if first_line:
                    num_taxa, seq_len = map(int, line.strip().split())
                    first_line = False
                    continue
                if not line:
                    continue

                if len(seq_names) < num_taxa:
                    temp_line = line.split()
                    seq_name = temp_line[0]
                    seq = "".join(temp_line[1:])

                    seq_names.append(seq_name)
                else:
                    temp_line = line.split()
                    seq = "".join(temp_line)

                sequence_dict[seq_names[counter]] += seq

                counter = (counter + 1) % num_taxa

        if len(seq_names) != num_taxa:
            raise ValueError(f"unexpected number of sequences: {len(seq_names)} vs {num_taxa}")

        for i in range(len(seq_names)):
            taxon = seq_names[i]
            sequence = sequence_dict[taxon]
            if len(sequence) != seq_len:
                raise ValueError(f"unexpected sequence length: {len(sequence)} vs {seq_len}")
            sequences.append(Sequence(taxon, sequence))

        return sequences


def parse_msa_somehow(msa_path):
    parsers = [
        FastaParser, PhylipsParser, PhylipParser
    ]
    num_successful = 0
    sequences = []
    sequences_dict = {}

    error_log = ""

    for parser_class in parsers:
        try:
            parser = parser_class()
            sequences = parser.parse(msa_path)
            sequences_dict[str(parser)] = sequences

            num_successful += 1
            break
        except Exception as e:
            error_log += f"{e}\n"

    if num_successful == 0:
        print(f"failed to parse {msa_path}")
        print(error_log)

    return sequences


def traverse_and_count(clade, msa, mutation_counter):
    """
    Traverses a tree recursively and counts the number of substitutions based on the parsimony rule.
    :param clade: current tree node
    :param msa: dict with map of leaf name to nucleotide sequence
    :param mutation_counter: list with per-site substitution counts
    :return:
    """
    if clade.name:
        return [[char] for char in msa[clade.name]]

    pars_seq = []
    for c in clade.clades:
        if not pars_seq:
            pars_seq = traverse_and_count(c, msa, mutation_counter)
        else:
            temp_pars_seq = traverse_and_count(c, msa, mutation_counter)
            for i in range(len(temp_pars_seq)):
                site1 = pars_seq[i]
                site2 = temp_pars_seq[i]

                isec = list(set(site1) & set(site2))
                if not isec:
                    mutation_counter[i] += 1

                if not isec:
                    isec = site1 + site2
                pars_seq[i] = isec
    return pars_seq


def count_subst_freqs(tree_path, msa_path):
    mutation_counter = collections.defaultdict(lambda: 0)

    with open(tree_path) as file:
        for line in file:
            handle = StringIO(line)
    tree = Phylo.read(handle, "newick")

    msa = parse_msa_somehow(msa_path)
    msa_dict = dict((seq.id, seq.sequence) for seq in msa)

    root_seq = traverse_and_count(tree.root, msa_dict, mutation_counter)
    mutation_counter_list = [mutation_counter[i] for i in range(len(msa_dict[list(msa_dict.keys())[0]]))]

    return mutation_counter_list


def example(tree_path, msa_path):
    print(count_subst_freqs(tree_path, msa_path))


def main():
    current_loo_targets = pd.read_csv(os.path.join(os.pardir, "data/processed/target/loo_result_entropy.csv"))
    dataset_set = set(current_loo_targets['dataset'])

    counter = 0

    for dataset in dataset_set:
        results = []

        counter += 1
        print(str(counter) + "/" + str(len(dataset_set)))
        print(os.path.join(os.pardir, "data/processed/features", dataset + "_subst_freq_stats.csv"))
        if os.path.exists(os.path.join(os.pardir, "data/processed/features", dataset + "_subst_freq_stats.csv")):
            print("Skipped, already exists " + dataset)
            #continue

        print("Dataset: " + dataset)
        reference_msa_path = os.path.join(os.pardir, "data/raw/msa", dataset + "_reference.fasta")
        reference_msa = AlignIO.read(reference_msa_path, 'fasta')
        reference_tree_path = os.path.join(os.pardir, "data/raw/reference_tree", dataset + ".newick")
        sample_df = current_loo_targets[current_loo_targets['dataset'] == dataset]["sampleId"]
        for sample in sample_df:
            # Split MSA into query and rest
            new_alignment = []
            for record in reference_msa:
                if record.id != sample:
                    seq_record = SeqIO.SeqRecord(seq=record.seq, id=record.id, description="")
                    new_alignment.append(seq_record)

            output_file_msa = os.path.join(os.pardir, "data/processed/loo", dataset + "_msa_" + sample + ".fasta")
            output_file_msa = os.path.abspath(output_file_msa)

            SeqIO.write(new_alignment, output_file_msa, "fasta")

            with open(reference_tree_path, 'r') as original_file:

                original_newick_tree = original_file.read()
                original_tree = ete3.Tree(original_newick_tree)

                leaf_names = original_tree.get_leaf_names()

                leaf_count = len(leaf_names)
                leaf_node = original_tree.search_nodes(name=sample)[0]

                leaf_node.delete()
                leaf_names = original_tree.get_leaf_names()
                output_file_tree = os.path.join(os.pardir, "data/raw/reference_tree", dataset + sample + ".newick")
                output_file_tree = os.path.abspath(output_file_tree)

                original_tree.write(outfile=output_file_tree, format=5)

                result = count_subst_freqs(output_file_tree, output_file_msa)

                max_subst_freq = max(result) / leaf_count
                avg_subst_freq = (sum(result) / len(result)) / leaf_count
                if statistics.mean(result) != 0:
                    cv_subst_freq = statistics.stdev(result) / statistics.mean(result)
                else:
                    cv_subst_freq = 0
                if max(result) == min(result):
                    normalized_result = result
                else:
                    normalized_result = [(x - min(result)) / (max(result) - min(result)) for x in
                                              result]
                sk_subst_freq = skew(normalized_result)
                kur_subst_freq = kurtosis(result, fisher=True)

                results.append(
                    (dataset, sample, max_subst_freq, avg_subst_freq, cv_subst_freq, sk_subst_freq, kur_subst_freq))

                os.remove(output_file_msa)
                os.remove(output_file_tree)

        df = pd.DataFrame(results,
                          columns=['dataset', 'sampleId', "max_subst_freq", "avg_subst_freq", "cv_subst_freq",
                                   "sk_subst_freq", "kur_subst_freq"])
        df.to_csv(os.path.join(os.pardir, "data/processed/features", dataset + "_subst_freq_stats.csv"))


if __name__ == "__main__":
    main()
