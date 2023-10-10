import logging
import os
import shutil
import statistics
import subprocess
import time
from collections import Counter

import numpy as np
from Bio import AlignIO, SeqRecord, Seq, SeqIO
from Bio.Align import MultipleSeqAlignment
from ete3 import Tree
from scipy.stats import entropy


class FeatureComputer:
    def __init__(self, msa_filepath, model_filepath, tree_filepath):
        self.logger = self.setup_logger()
        self.msa_filepath = msa_filepath
        self.model_filepath = model_filepath
        self.tree_filepath = tree_filepath

    def setup_logger(self):
        logger = logging.getLogger('FeatureComputer')
        logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        logger.addHandler(ch)

        return logger

    def compute_split_features(self):
        with open(self.tree_filepath, "r") as support_file:
            tree_str = support_file.read()
            phylo_tree = Tree(tree_str)
            branch_id_counter = 0

            for node in phylo_tree.traverse():
                branch_id_counter += 1
                if not node.is_leaf():
                    node.__setattr__("name", branch_id_counter)

            results = []

            for node in phylo_tree.traverse("postorder"):
                if (not node.is_root()) and (not node.is_leaf()):

                    list_a = []
                    list_a_dist_branch = []
                    list_a_dist_topo = []
                    list_b = []
                    list_b_dist_branch = []
                    list_b_dist_topo = []
                    for leaf in phylo_tree.get_leaves():
                        if leaf in node.get_leaves():
                            list_a.append(leaf.name)
                            list_a_dist_branch.append(leaf.get_distance(target=phylo_tree.get_tree_root()))
                            list_a_dist_topo.append(
                                leaf.get_distance(topology_only=True, target=phylo_tree.get_tree_root()))
                        else:
                            list_b.append(leaf.name)
                            list_b_dist_branch.append(leaf.get_distance(target=phylo_tree.get_tree_root()))
                            list_b_dist_topo.append(
                                leaf.get_distance(topology_only=True, target=phylo_tree.get_tree_root()))


                    split_mean_dist_branch_a = statistics.mean(list_a_dist_branch)
                    split_std_dist_branch_a = np.std(list_a_dist_branch)

                    split_mean_dist_branch_b = statistics.mean(list_b_dist_branch)
                    split_std_dist_branch_b = np.std(list_b_dist_branch)

                    split_std_ratio_branch = min(split_std_dist_branch_a, split_std_dist_branch_b) / max(
                        split_std_dist_branch_a, split_std_dist_branch_b)
                    split_mean_ratio_branch = min(split_mean_dist_branch_a, split_mean_dist_branch_b) / max(
                        split_mean_dist_branch_a, split_mean_dist_branch_b)

                    split_std_dist_topo_a = np.std(list_a_dist_topo)

                    split_std_dist_topo_b = np.std(list_b_dist_topo)

                    split_std_ratio_topo = min(split_std_dist_topo_a, split_std_dist_topo_b) / max(
                        split_std_dist_topo_a, split_std_dist_topo_b)

                    alignment = AlignIO.read(self.msa_filepath, 'fasta')
                    alignment_a = MultipleSeqAlignment([])
                    alignment_b = MultipleSeqAlignment([])
                    for record in alignment:
                        if record.id in list_a:
                            alignment_a.append(record)
                        elif record.id in list_b:
                            alignment_b.append(record)

                    freqs_b = []
                    freqs_a = []

                    entropy_differences = []

                    for i in range(len(alignment_a[0])):
                        column_a = alignment_a[:, i]

                        column_b = alignment_b[:, i]

                        combined_values = column_a + column_b
                        all_keys = set(combined_values)

                        counter_a = Counter({key: 0 for key in all_keys})
                        counter_b = Counter({key: 0 for key in all_keys})

                        counter_a.update(column_a)
                        counter_b.update(column_b)

                        sorted_keys = sorted(all_keys)

                        counter_a = Counter({key: counter_a[key] for key in sorted_keys})
                        counter_b = Counter({key: counter_b[key] for key in sorted_keys})

                        freqs_a.append(counter_a)
                        freqs_b.append(counter_b)

                    for site_freq_a, site_freq_b in zip(freqs_a, freqs_b):
                        total_count_a = sum(site_freq_a.values())
                        total_count_b = sum(site_freq_b.values())
                        try:
                            normalized_freq_a = {k: v / total_count_a for
                                                 k, v in site_freq_a.items()}
                        except ZeroDivisionError:
                            normalized_freq_a = {k: 0 for
                                                 k, v in site_freq_a.items()}
                        try:
                            normalized_freq_b = {k: v / total_count_b for
                                                 k, v in site_freq_b.items()}
                        except:
                            normalized_freq_b = {k: 0 for
                                                 k, v in site_freq_b.items()}

                        site_freq_a_array = np.array(list(normalized_freq_a.values()))

                        site_freq_b_array = np.array(list(normalized_freq_b.values()))

                        entropy_a = entropy(site_freq_a_array)
                        entropy_b = entropy(site_freq_b_array)

                        entropy_difference = abs(entropy_a - entropy_b)
                        entropy_differences.append(entropy_difference)

                    split_std_entropy_diff = np.std(entropy_differences)

                    result = (node.name,
                              split_std_entropy_diff,
                              split_std_ratio_topo,
                                split_mean_ratio_branch
                              , split_std_ratio_branch)
                    results.append(result)
            return results



    def compute_parsimony_support_features(self, support_path):

        with open(support_path, "r") as support_file:
            tree_str = support_file.read()
            tree = Tree(tree_str)
            branch_id_counter = 0
            results = []
            farthest_branch = tree.get_farthest_leaf(topology_only=False)[1]
            for node in tree.traverse():
                branch_id_counter += 1
                node.__setattr__("name", branch_id_counter)
                length = node.dist
                length_relative = length / farthest_branch
                if node.support is not None and not node.is_leaf():

                    childs_inner = [node_child for node_child in node.traverse() if not node_child.is_leaf()]
                    parents_inner = node.get_ancestors()
                    supports_childs = []
                    weighted_supports_childs = []
                    for child in childs_inner:
                        supports_childs.append(child.support)
                        weighted_supports_childs.append(child.support * child.dist)

                    supports_parents = []
                    weighted_supports_parents = []
                    for parent in parents_inner:
                        supports_parents.append(parent.support)
                        weighted_supports_parents.append(parent.support * parent.dist)

                    if len(weighted_supports_childs) >= 1:
                        min_pars_supp_child_w = min(weighted_supports_childs)
                        max_pars_supp_child_w = max(weighted_supports_childs)
                    else:
                        min_pars_supp_child_w = -1
                        max_pars_supp_child_w = -1

                    if len(weighted_supports_parents) >= 1:
                        mean_pars_supp_parents_w = statistics.mean(weighted_supports_parents)
                    else:
                        mean_pars_supp_parents_w = -1

                    results.append((node.name, node.support / 100,
                                    min_pars_supp_child_w,
                                    max_pars_supp_child_w, mean_pars_supp_parents_w, length, length_relative
                                    ))
        return results

    def compute_parsimony_support(self):
        start_time = time.time()

        output_prefix = "parsimony_tmp_1000"

        raxml_command = [
            "raxml-ng",
            "--start",
            f"--model {self.model_filepath}",
            "--tree pars{1000}",
            f"--msa {self.msa_filepath}",
            "--redo",
            f"--prefix {output_prefix}"
        ]

        subprocess.run(" ".join(raxml_command), shell=True)

        parsimonies_filepath = os.path.join(os.curdir, output_prefix + ".raxml.startTree")
        raxml_command = ["raxml-ng",
                         "--support",
                         f"--tree {self.tree_filepath}",
                         f"--bs-trees {parsimonies_filepath}",
                         "--redo",
                         f"--prefix {output_prefix}"]

        subprocess.run(" ".join(raxml_command), shell=True)

        return self.compute_parsimony_bootstrap_support_features(
            os.path.abspath(parsimonies_filepath.replace("startTree", "support")))

    def compute_parsimony_bootstrap_support_features(self, support_path):

        results = []

        with open(support_path, "r") as support_file:
            tree_str = support_file.read()
            tree = Tree(tree_str)

            all_supports = []
            for node in tree.traverse():
                if not node.is_leaf():
                    all_supports.append(node.support)

            branch_id_counter = 0
            for node in tree.traverse():
                branch_id_counter += 1
                node.__setattr__("name", branch_id_counter)
                if node.support is not None and not node.is_leaf():

                    parents_inner = node.get_ancestors()

                    supports_parents = []
                    for parent in parents_inner:
                        supports_parents.append(parent.support)

                    if len(supports_parents) >= 1:
                        mean_pars_bootsupp_parents = statistics.mean(supports_parents)
                        std_pars_bootsupp_parents = np.std(supports_parents)
                    else:
                        mean_pars_bootsupp_parents = -1
                        std_pars_bootsupp_parents = -1

                    results.append(
                        (node.name, node.support / 100, mean_pars_bootsupp_parents, std_pars_bootsupp_parents))
        self.logger.info("Finished computing Parsimony Bootstrap features")

        print(results)
        return results

    def compute_parsimony_bootstrap_support(self):

        start_time = time.time()

        alignment = AlignIO.read(self.msa_filepath, "fasta")
        sequence_data = [list(record.seq) for record in alignment]
        alignment_array = np.array(sequence_data)
        original_ids = [record.id for record in alignment]

        tmp_folder_path = os.path.join(os.curdir, "tmp")

        if os.path.exists(tmp_folder_path):
            shutil.rmtree(tmp_folder_path)

        os.makedirs(tmp_folder_path)
        os.chdir(tmp_folder_path)

        trees_path = os.path.join(os.curdir, "parsimony_bootstraps_tmp.txt")

        for x in range(200):
            sampled_columns = np.random.choice(alignment_array.shape[1], size=alignment_array.shape[1],
                                               replace=True)
            replicate_alignment = alignment_array[:, sampled_columns]

            seq_records = [SeqRecord.SeqRecord(Seq.Seq(''.join(seq)), id=original_ids[i], description="") for i, seq
                           in
                           enumerate(replicate_alignment)]

            msa_new = AlignIO.MultipleSeqAlignment(seq_records)

            new_msa_path = os.path.join(os.curdir, "parsimony_bootstrap_tmp_" + str(x) + ".fasta")

            output_prefix = "parsimony_bootstrap_tmp_" + str(
                x)

            SeqIO.write(msa_new, new_msa_path, "fasta")

            raxml_path = subprocess.check_output(["which", "raxml-ng"], text=True).strip()

            raxml_command = [
                raxml_path,
                "--start",
                f"--model {self.model_filepath}",
                "--tree pars{1}",
                f"--msa {new_msa_path}",
                "--redo",
                "--log ERROR",
                f"--prefix {output_prefix}",
            ]
            try:
                subprocess.run(" ".join(raxml_command), shell=True)
            except:
                self.logger.error()

            result_tree_path = os.path.join(os.curdir, output_prefix + ".raxml.startTree")

            try:
                with open(os.path.abspath(result_tree_path), 'r') as tree_file:
                    newick_tree = tree_file.read()
            except FileNotFoundError:
                print("boot tree not found")
                continue
            # Append the Newick tree to the trees file
            with open(trees_path, 'a') as trees_file:
                # If the file doesn't exist, create it and add the tree
                if not os.path.exists(os.path.abspath(trees_path)):
                    trees_file.write(newick_tree)
                    print("added tree")
                else:
                    # Append the tree
                    trees_file.write(newick_tree)
        self.logger.info("Finished computing 200 Parsimony Bootstraps, begin feature extraction")
        raxml_command = ["raxml-ng",
                         "--support",
                         f"--tree {self.tree_filepath}",
                         f"--bs-trees {trees_path}",
                         "--redo",
                         f"--prefix {output_prefix}",
                         "--log ERROR"
                         ]
        subprocess.run(" ".join(raxml_command), shell=True)

        end_time = time.time()
        elapsed_time = end_time - start_time

        logging.info("Computed 200 Parsimony Bootstraps ... ")
        logging.info("Elpased time (seconds):", elapsed_time)

        return self.compute_parsimony_bootstrap_support_features(
            os.path.join(os.path.abspath(os.curdir), "parsimony_bootstrap_tmp_199.raxml.support"))


if __name__ == "__main__":
    feature_computer = FeatureComputer(
        "/Users/juliuswiegert/Repositories/placement_difficulty_prediction/data/raw/msa/138_0_reference.fasta",
        "/Users/juliuswiegert/Repositories/placement_difficulty_prediction/data/processed/loo/138_0_msa_model.txt",
        "/Users/juliuswiegert/Repositories/placement_difficulty_prediction/data/raw/reference_tree/138_0.newick")

    data = "your_data"
    #feature1_result = feature_computer.compute_parsimony_bootstrap_support()
    #feature2_result = feature_computer.compute_parsimony_support()
    feature1_result = feature_computer.compute_split_features()
    print(feature1_result)



#df = df[["", "", "", "", "", "avg_subst_freq",
 #        "", "max_subst_freq", "avg_rel_rf_boot", "", "",
  #       "", "", "cv_subst_freq",
   #      "",
    #     "", "", "bl_ratio", "",
     #    "mean_clo_sim_ratio", "", ""
      #   ]]