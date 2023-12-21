import multiprocessing
import random
import types
from collections import Counter

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
import numpy as np
from scipy.stats import kurtosis, skew


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
    alignment_original = AlignIO.read(filepath, 'fasta')
    for record in alignment_original:
        alignment = [record_ for record_ in alignment_original if record_.id != record.id]









        dataset = query_filepath.replace("_query.fasta", "")

        likpath = os.path.join(os.pardir, "scripts/", query_filepath.replace("_query.fasta", "") + "_siteliks_.raxml.siteLH")
        try:
            with open(likpath, 'r') as file:
                # Read the lines from the file
                lines = file.readlines()

            # Check if there are at least two lines in the file
            if len(lines) >= 2:
                # Extract the second line (index 1) and remove leading/trailing whitespace
                second_line = lines[1].strip()
                # print(second_line)
                # Use regular expression to extract numbers from the second line
                import re

                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", second_line)
                numbers = numbers[1:]

                numbers = [float(number) for number in numbers]
                min_value = min(numbers)
                max_value = max(numbers)

                # Perform min-max scaling
                scaled_numbers = [(x / sum(numbers)) for x in numbers]

                threshold_9 = sorted(scaled_numbers)[int(len(scaled_numbers) * 0.9)]
                analyzed_sites_9 = [1 if x <= threshold_9 else 0 for x in scaled_numbers]

                threshold_8 = sorted(scaled_numbers)[int(len(scaled_numbers) * 0.8)]
                analyzed_sites_8= [1 if x <= threshold_8 else 0 for x in scaled_numbers]

                threshold_95 = sorted(scaled_numbers)[int(len(scaled_numbers) * 0.95)]
                analyzed_sites_95 = [1 if x <= threshold_95 else 0 for x in scaled_numbers]

                threshold_7 = sorted(scaled_numbers)[int(len(scaled_numbers) * 0.7)]
                analyzed_sites_7 = [1 if x <= threshold_7 else 0 for x in scaled_numbers]

                threshold_3 = sorted(scaled_numbers)[int(len(scaled_numbers) * 0.3)]
                analyzed_sites_3 = [1 if x <= threshold_3 else 0 for x in scaled_numbers]

                threshold_1 = sorted(scaled_numbers)[int(len(scaled_numbers) * 0.1)]
                analyzed_sites_1 = [1 if x <= threshold_1 else 0 for x in scaled_numbers]

                threshold_6 = sorted(scaled_numbers)[int(len(scaled_numbers) * 0.6)]
                analyzed_sites_6 = [1 if x <= threshold_6 else 0 for x in scaled_numbers]

                threshold_5 = sorted(scaled_numbers)[int(len(scaled_numbers) * 0.5)]
                analyzed_sites_5 = [1 if x <= threshold_5 else 0 for x in scaled_numbers]

                threshold_4 = sorted(scaled_numbers)[int(len(scaled_numbers) * 0.4)]
                analyzed_sites_4 = [1 if x <= threshold_4 else 0 for x in scaled_numbers]

                threshold_2 = sorted(scaled_numbers)[int(len(scaled_numbers) * 0.2)]
                analyzed_sites_2 = [1 if x <= threshold_2 else 0 for x in scaled_numbers]
        except:
            print(likpath)
            return -1

        try:
            for position in range(len(alignment[0])):
                break
        except IndexError:
            return -1
        # Iterate over each position in the alignment

        isAA = False
        loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
        datatype = loo_selection[loo_selection["verbose_name"] == query_filepath.replace("_query.fasta", ".phy")].iloc[0][
            "data_type"]
        if datatype == "AA" or datatype == "DataType.AA":
            isAA = True
            print("Found AA")
        gap_matches = 0
        total_gap_count = 0

        if len(analyzed_sites_8) == 0:
            print("Error")
            return -1

        for i, (flag, char) in enumerate(analyzed_sites_8):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 0 and char in ["-", "N"]:
                total_gap_count += 1
                if str(record.seq)[i] == char:
                    gap_matches += 1

        match_counter_9 = 0
        total_inv_sites_9 = 0
        for i, (flag, char) in enumerate(analyzed_sites_9):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 1:
                total_inv_sites_9 += 1
            if flag == 1 and str(record.seq)[i] == char and char not in ['-', 'N']:
                match_counter_9 += 1

        match_counter_6 = 0
        total_inv_sites_6 = 0
        for i, (flag, char) in enumerate(analyzed_sites_6):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 1:
                total_inv_sites_6 += 1
            if flag == 1 and str(record.seq)[i] == char and char not in ['-', 'N']:
                match_counter_6 += 1

        match_counter_5 = 0
        total_inv_sites_5 = 0
        for i, (flag, char) in enumerate(analyzed_sites_5):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 1:
                total_inv_sites_5 += 1
            if flag == 1 and str(record.seq)[i] == char and char not in ['-', 'N']:
                match_counter_5 += 1

        match_counter_4 = 0
        total_inv_sites_4 = 0
        for i, (flag, char) in enumerate(analyzed_sites_4):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 1:
                total_inv_sites_4 += 1
            if flag == 1 and str(record.seq)[i] == char and char not in ['-', 'N']:
                match_counter_4 += 1

        match_counter_2 = 0
        total_inv_sites_2 = 0
        for i, (flag, char) in enumerate(analyzed_sites_2):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 1:
                total_inv_sites_2 += 1
            if flag == 1 and str(record.seq)[i] == char and char not in ['-', 'N']:
                match_counter_2 += 1

        match_counter_8 = 0
        total_inv_sites_8 = 0
        for i, (flag, char) in enumerate(analyzed_sites_8):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 1:
                total_inv_sites_8 += 1
            if flag == 1 and str(record.seq)[i] == char and char not in ['-', 'N']:
                match_counter_8 += 1

        transition_count8 = 0
        transversion_count8 = 0
        mut_count8 = 0
        fraction_char_rests8 = []
        for i, (flag, char) in enumerate(analyzed_sites_8):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 1:
                total_inv_sites_8 += 1
            #if flag == 1 and str(record.seq)[i] == char and char not in ['-', 'N']:
             #   match_counter_8 += 1
            if flag == 1 and str(record.seq)[i] != char:
                mut_count8 += 1
                if char in ["C", "T", "U"]:
                    if str(record.seq)[i] in ["A", "G"]:
                        transversion_count8 += 1
                elif char in ["A", "G"]:
                    if str(record.seq)[i] in ["C", "T", "U"]:
                        transversion_count8 += 1
                else:
                    transition_count8 += 1

                residues_at_position = [str(record.seq[i]) for record in alignment]
                residue_counts = Counter(residues_at_position)
                most_common_residue, most_common_count = residue_counts.most_common(1)[0]
                residues_at_position_del_most_common = [r for r in residues_at_position if r != most_common_residue]
                if str(record.seq)[i] in residues_at_position_del_most_common:
                    count_char = residues_at_position_del_most_common.count(str(record.seq)[i])
                    fraction_char_rest = count_char / len(residues_at_position_del_most_common)
                else:
                    fraction_char_rest = 0
                fraction_char_rests8.append(fraction_char_rest)


        match_counter_7 = 0
        total_inv_sites_7 = 0
        for i, (flag, char) in enumerate(analyzed_sites_7):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 1:
                total_inv_sites_7 += 1
            if flag == 1 and str(record.seq)[i] == char and char not in ['-', 'N']:
                match_counter_7 += 1

        transition_count7 = 0
        transversion_count7 = 0
        mut_count7 = 0
        fraction_char_rests7 = []
        for i, (flag, char) in enumerate(analyzed_sites_7):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 1:
                total_inv_sites_7 += 1
            #if flag == 1 and str(record.seq)[i] == char and char not in ['-', 'N']:
             #   match_counter_7 += 1
            if flag == 1 and str(record.seq)[i] != char:
                mut_count7 += 1
                if char in ["C", "T", "U"]:
                    if str(record.seq)[i] in ["A", "G"]:
                        transversion_count7 += 1
                elif char in ["A", "G"]:
                    if str(record.seq)[i] in ["C", "T", "U"]:
                        transversion_count7 += 1
                else:
                    transition_count7 += 1

                residues_at_position = [str(record.seq[i]) for record in alignment]
                residue_counts = Counter(residues_at_position)
                most_common_residue, most_common_count = residue_counts.most_common(1)[0]
                residues_at_position_del_most_common = [r for r in residues_at_position if r != most_common_residue]
                if str(record.seq)[i] in residues_at_position_del_most_common:
                    count_char = residues_at_position_del_most_common.count(str(record.seq)[i])
                    fraction_char_rest = count_char / len(residues_at_position_del_most_common)
                else:
                    fraction_char_rest = 0
                fraction_char_rests7.append(fraction_char_rest)

        transition_count5 = 0
        transversion_count5 = 0
        mut_count5 = 0
        fraction_char_rests5 = []
        for i, (flag, char) in enumerate(analyzed_sites_5):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 1:
                total_inv_sites_5 += 1
            #if flag == 1 and str(record.seq)[i] == char and char not in ['-', 'N']:
             #   match_counter_5 += 1
            if flag == 1 and str(record.seq)[i] != char:
                mut_count5 += 1
                if char in ["C", "T", "U"]:
                    if str(record.seq)[i] in ["A", "G"]:
                        transversion_count5 += 1
                elif char in ["A", "G"]:
                    if str(record.seq)[i] in ["C", "T", "U"]:
                        transversion_count5 += 1
                else:
                    transition_count5 += 1

                residues_at_position = [str(record.seq[i]) for record in alignment]
                residue_counts = Counter(residues_at_position)
                most_common_residue, most_common_count = residue_counts.most_common(1)[0]
                residues_at_position_del_most_common = [r for r in residues_at_position if r != most_common_residue]
                if str(record.seq)[i] in residues_at_position_del_most_common:
                    count_char = residues_at_position_del_most_common.count(str(record.seq)[i])
                    fraction_char_rest = count_char / len(residues_at_position_del_most_common)
                else:
                    fraction_char_rest = 0
                fraction_char_rests5.append(fraction_char_rest)

        transition_count9 = 0
        transversion_count9 = 0
        mut_count9 = 0
        fraction_char_rests9 = []
        for i, (flag, char) in enumerate(analyzed_sites_9):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 1:
                total_inv_sites_9 += 1
            #if flag == 1 and str(record.seq)[i] == char and char not in ['-', 'N']:
             #   match_counter_9 += 1
            if flag == 1 and str(record.seq)[i] != char:
                mut_count9 += 1
                if char in ["C", "T", "U"]:
                    if str(record.seq)[i] in ["A", "G"]:
                        transversion_count9 += 1
                elif char in ["A", "G"]:
                    if str(record.seq)[i] in ["C", "T", "U"]:
                        transversion_count9 += 1
                else:
                    transition_count9 += 1

                residues_at_position = [str(record.seq[i]) for record in alignment]
                residue_counts = Counter(residues_at_position)
                most_common_residue, most_common_count = residue_counts.most_common(1)[0]
                residues_at_position_del_most_common = [r for r in residues_at_position if r != most_common_residue]
                if str(record.seq)[i] in residues_at_position_del_most_common:
                    count_char = residues_at_position_del_most_common.count(str(record.seq)[i])
                    fraction_char_rest = count_char / len(residues_at_position_del_most_common)
                else:
                    fraction_char_rest = 0
                fraction_char_rests9.append(fraction_char_rest)


        match_counter_95 = 0
        total_inv_sites_95 = 0
        for i, (flag, char) in enumerate(analyzed_sites_95):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 1:
                total_inv_sites_95 += 1
            if flag == 1 and str(record.seq)[i] == char and char not in ['-', 'N']:
                match_counter_95 += 1

        match_counter_1 = 0
        total_inv_sites_1 = 0
        for i, (flag, char) in enumerate(analyzed_sites_1):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 1:
                total_inv_sites_1 += 1
            if flag == 1 and str(record.seq)[i] == char and char not in ['-', 'N']:
                match_counter_1 += 1

        match_counter_3 = 0
        total_inv_sites_3 = 0
        for i, (flag, char) in enumerate(analyzed_sites_3):
            # Check if the corresponding site in the query has a 1 and if the characters are equal
            if flag == 1:
                total_inv_sites_3 += 1
            if flag == 1 and str(record.seq)[i] == char and char not in ['-', 'N']:
                match_counter_3 += 1

        sequence = str(record.seq)
        seq_length = len(sequence)

        name = ""

        if query_filepath == "neotrop_query_10k.fasta":
            name = "neotrop"
        elif query_filepath == "bv_query.fasta":
            name = "bv"
        elif query_filepath == "tara_query.fasta":
            name = "tara"
        else:
            name = query_filepath.replace("_query.fasta", "")

        if total_inv_sites_7 > 0:
            match_rel_7 = match_counter_7 / total_inv_sites_7
        else:
            match_rel_7 = 0

        if total_inv_sites_8 > 0:
            match_rel_8 = match_counter_8 / total_inv_sites_8
        else:
            match_rel_8 = 0

        if total_inv_sites_9 > 0:
            match_rel_9 = match_counter_9 / total_inv_sites_9
        else:
            match_rel_9 = 0

        if total_inv_sites_95 > 0:
            match_rel_95 = match_counter_95 / total_inv_sites_95
        else:
            match_rel_95 = 0

        if total_inv_sites_3 > 0:
            match_rel_3 = match_counter_3 / total_inv_sites_3
        else:
            match_rel_3 = 0

        if total_inv_sites_6 > 0:
            match_rel_6 = match_counter_6 / total_inv_sites_6
        else:
            match_rel_6 = 0

        if total_inv_sites_5 > 0:
            match_rel_5 = match_counter_5 / total_inv_sites_5
        else:
            match_rel_5 = 0

        if total_inv_sites_4 > 0:
            match_rel_4 = match_counter_4 / total_inv_sites_4
        else:
            match_rel_4 = 0

        if total_inv_sites_2 > 0:
            match_rel_2 = match_counter_2 / total_inv_sites_2
        else:
            match_rel_2 = 0

        if total_inv_sites_1 > 0:
            match_rel_1 = match_counter_1 / total_inv_sites_1
        else:
            match_rel_1 = 0

        if total_gap_count > 0:
            match_rel_gap = gap_matches / total_gap_count
        else:
            match_rel_gap = 0

        if mut_count8 > 0:
            transition_count_rel8 = transition_count8 / mut_count8
            transversion_count_rel8 = transversion_count8 / mut_count8
        else:
            transition_count_rel8 = 0
            transversion_count_rel8 = 0

        if len(fraction_char_rests8) > 0:
            max_fraction_char_rests8 = np.max(fraction_char_rests8)
            min_fraction_char_rests8 = np.min(fraction_char_rests8)
            avg_fraction_char_rests8 = np.mean(fraction_char_rests8)
            std_fraction_char_rests8 = np.std(fraction_char_rests8)
            skw_fraction_char_rests8 = skew(fraction_char_rests8)
            kur_fraction_char_rests8 = kurtosis(fraction_char_rests8, fisher=True)
        else:
            max_fraction_char_rests8 = -1
            min_fraction_char_rests8 = -1
            avg_fraction_char_rests8 = -1
            std_fraction_char_rests8 = -1
            skw_fraction_char_rests8 = -1
            kur_fraction_char_rests8 = -1

        if mut_count7 > 0:
            transition_count_rel7 = transition_count7 / mut_count7
            transversion_count_rel7 = transversion_count7 / mut_count7
        else:
            transition_count_rel7 = 0
            transversion_count_rel7 = 0

        if len(fraction_char_rests7) > 0:
            max_fraction_char_rests7 = np.max(fraction_char_rests7)
            min_fraction_char_rests7 = np.min(fraction_char_rests7)
            avg_fraction_char_rests7 = np.mean(fraction_char_rests7)
            std_fraction_char_rests7 = np.std(fraction_char_rests7)
            skw_fraction_char_rests7 = skew(fraction_char_rests7)
            kur_fraction_char_rests7 = kurtosis(fraction_char_rests7, fisher=True)
        else:
            max_fraction_char_rests7 = -1
            min_fraction_char_rests7 = -1
            avg_fraction_char_rests7 = -1
            std_fraction_char_rests7 = -1
            skw_fraction_char_rests7 = -1
            kur_fraction_char_rests7 = -1

        if mut_count5 > 0:
            transition_count_rel5 = transition_count5 / mut_count5
            transversion_count_rel5 = transversion_count5 / mut_count5
        else:
            transition_count_rel5 = 0
            transversion_count_rel5 = 0

        if len(fraction_char_rests5) > 0:
            max_fraction_char_rests5 = np.max(fraction_char_rests5)
            min_fraction_char_rests5 = np.min(fraction_char_rests5)
            avg_fraction_char_rests5 = np.mean(fraction_char_rests5)
            std_fraction_char_rests5 = np.std(fraction_char_rests5)
            skw_fraction_char_rests5 = skew(fraction_char_rests5)
            kur_fraction_char_rests5 = kurtosis(fraction_char_rests5, fisher=True)
        else:
            max_fraction_char_rests5 = -1
            min_fraction_char_rests5 = -1
            avg_fraction_char_rests5 = -1
            std_fraction_char_rests5 = -1
            skw_fraction_char_rests5 = -1
            kur_fraction_char_rests5 = -1

        if mut_count9 > 0:
            transition_count_rel9 = transition_count9 / mut_count9
            transversion_count_rel9 = transition_count9 / mut_count9
        else:
            transition_count_rel9 = 0
            transversion_count_rel9 = 0

        if len(fraction_char_rests9) > 0:
            max_fraction_char_rests9 = np.max(fraction_char_rests9)
            min_fraction_char_rests9 = np.min(fraction_char_rests9)
            avg_fraction_char_rests9 = np.mean(fraction_char_rests9)
            std_fraction_char_rests9 = np.std(fraction_char_rests9)
            skw_fraction_char_rests9 = skew(fraction_char_rests9)
            kur_fraction_char_rests9 = kurtosis(fraction_char_rests9, fisher=True)
        else:
            max_fraction_char_rests9 = -1
            min_fraction_char_rests9 = -1
            avg_fraction_char_rests9 = -1
            std_fraction_char_rests9 = -1
            skw_fraction_char_rests9 = -1
            kur_fraction_char_rests9 = -1

        if isAA:
            transversion_count_rel5 = -1
            transversion_count_rel7 = -1
            transversion_count_rel8 = -1
            transversion_count_rel9 = -1

            transition_count_rel5 = -1
            transition_count_rel7 = -1
            transition_count_rel8 = -1
            transition_count_rel9 = -1

        results.append((name, record.id,
                        match_counter_7 / seq_length, match_counter_8 / seq_length, match_counter_9 / seq_length,
                        match_counter_95 / seq_length, match_counter_3 / seq_length, match_counter_1 / seq_length,
                        match_rel_7, match_rel_8, match_rel_9, match_rel_95, match_rel_3, match_rel_1, match_rel_gap,
                        match_rel_2, match_rel_4, match_rel_6, match_rel_5,
                        transition_count_rel9, transversion_count_rel9, max_fraction_char_rests9,
                        min_fraction_char_rests9, avg_fraction_char_rests9, std_fraction_char_rests9,
                        skw_fraction_char_rests9, kur_fraction_char_rests9,
                        transition_count_rel8, transversion_count_rel8, max_fraction_char_rests8,
                        min_fraction_char_rests8, avg_fraction_char_rests8, std_fraction_char_rests8,
                        skw_fraction_char_rests8, kur_fraction_char_rests8,
                        transition_count_rel7, transversion_count_rel7, max_fraction_char_rests7,
                        min_fraction_char_rests7, avg_fraction_char_rests7, std_fraction_char_rests7,
                        skw_fraction_char_rests7, kur_fraction_char_rests7,
                        transition_count_rel5, transversion_count_rel5, max_fraction_char_rests5,
                        min_fraction_char_rests5, avg_fraction_char_rests5, std_fraction_char_rests5,
                        skw_fraction_char_rests5, kur_fraction_char_rests5))

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
        if result != -1:
            results.append(result)
            print(counter)
            counter += 1

    pool.close()
    pool.join()

    results = [item for sublist in results for item in sublist]

    df = pd.DataFrame(results, columns=["dataset", "sampleId",
                        "match_counter_7_lik", "match_counter_8_lik", "match_counter_9_lik",
                        "match_counter_95_lik", "match_counter_3_lik", "match_counter_1_lik",
                        "match_rel_7_lik", "match_rel_8_lik", "match_rel_9_lik", "match_rel_95_lik", "match_rel_3_lik", "match_rel_1_lik", "match_rel_gap_lik",
                        "match_rel_2_lik", "match_rel_4_lik", "match_rel_6_lik", "match_rel_5_lik",
                        "transition_count_rel9_lik", "transversion_count_rel9_lik", "max_fraction_char_rests9_lik",
                        "min_fraction_char_rests9_lik", "avg_fraction_char_rests9_lik", "std_fraction_char_rests9_lik",
                        "skw_fraction_char_rests9_lik", "kur_fraction_char_rests9_lik",
                        "transition_count_rel8_lik", "transversion_count_rel8_lik", "max_fraction_char_rests8_lik",
                        "min_fraction_char_rests8_lik", "avg_fraction_char_rests8_lik", "std_fraction_char_rests8_lik",
                        "skw_fraction_char_rests8_lik", "kur_fraction_char_rests8_lik",
                        "transition_count_rel7_lik", "transversion_count_rel7_lik", "max_fraction_char_rests7_lik",
                        "min_fraction_char_rests7_lik", "avg_fraction_char_rests7_lik", "std_fraction_char_rests7_lik",
                        "skw_fraction_char_rests7_lik", "kur_fraction_char_rests7_lik",
                        "transition_count_rel5_lik", "transversion_count_rel5_lik", "max_fraction_char_rests5_lik",
                        "min_fraction_char_rests5_lik", "avg_fraction_char_rests5_lik", "std_fraction_char_rests5_lik",
                        "skw_fraction_char_rests5_lik", "kur_fraction_char_rests5_lik"])
    df.to_csv(os.path.join(os.pardir, "data/processed/features", "query_features_lik.csv"))
