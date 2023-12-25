import multiprocessing
import statistics
import sys
import types
import pylcs
import pandas as pd
import os
from Bio import SeqIO
from scipy.fftpack import dct
from collections import defaultdict
from scipy.stats import kurtosis, skew
import numpy as np
import cv2
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import normalize


def generate_k_mers(input_string, k):
    k_mers = []

    if len(input_string) < k:
        return k_mers

    for i in range(len(input_string) - k + 1):
        k_mer = input_string[i:i + k]
        k_mers.append(k_mer)

    return k_mers


def fraction_shared_kmers(binary_string1, binary_string2, k):
    kmers1 = generate_k_mers(binary_string1, k)
    kmers2 = generate_k_mers(binary_string2, k)

    set_kmers1 = set(kmers1)
    set_kmers2 = set(kmers2)

    shared_kmers = set_kmers1.intersection(set_kmers2)

    try:

        fraction_shared = len(shared_kmers) / (len(set_kmers1) + len(set_kmers2) - len(shared_kmers))
    except ZeroDivisionError:
        fraction_shared = 0

    return fraction_shared


def lbp_histogram(image):
    patterns = local_binary_pattern(image, 8, 1)
    hist, _ = np.histogram(patterns, bins=np.arange(2 ** 8 + 1), density=True)
    return hist


def calculate_hu_moments(contour):
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments)

    return hu_moments.flatten()


def calculate_shape_similarity(hu_moments1, hu_moments2):
    return np.linalg.norm(hu_moments1 - hu_moments2)


def dna_to_numeric(sequence, isAA):
    if not isAA:
        mapping = {'A': 63, 'C': 127, 'G': 191, 'T': 255, '-': 0, 'N': 0}
        mapping = defaultdict(lambda: 0, mapping)
    else:
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        step = 256 // len(amino_acids)
        # Create a linear mapping with evenly distributed grayscale values
        mapping = {aa: i * step for i, aa in enumerate(amino_acids)}
        mapping = defaultdict(lambda: 0, mapping)
    numeric_sequence = [mapping[base] for base in sequence]
    return np.array(numeric_sequence)


def encode_dna_as_image(sequence):
    # Calculate the nearest square side length
    side_length = int(np.ceil(np.sqrt(len(sequence))))
    image = np.resize(sequence, (side_length, side_length))
    return image


def compute_hamming_distance(hash_value_1, hash_value_2):
    distance = sum(c1 != c2 for c1, c2 in zip(hash_value_1, hash_value_2))
    return distance


def compute_dct_sign_only_hash(sequence, isAA):
    numeric_sequence = dna_to_numeric(sequence, isAA)
    image = encode_dna_as_image(numeric_sequence)

    dct_coeffs = dct(dct(image, axis=0), axis=1)
    sign_only_sequence = np.sign(dct_coeffs)
    size_ = 16

    try:
        # sign_only_sequence = sign_only_sequence[np.ix_(list(range(size_)),
        #                                               list(range(size_)))]
        # hash_value = "".join([str(int(sign)) for sign in sign_only_sequence.flatten()])

        sign_only_sequence[sign_only_sequence >= 0] = 1
        sign_only_sequence[sign_only_sequence < 0] = 0
        binary_sequence = sign_only_sequence[:size_, :size_]

        # Flatten the binary matrix into a binary string
        hash_value = "".join([str(int(bit)) for bit in binary_sequence.flatten()])
        normalized_coeff = (dct_coeffs[:size_, :size_] - np.min(dct_coeffs[:size_, :size_])) / (
                    np.max(dct_coeffs[:size_, :size_]) - np.min(dct_coeffs[:size_, :size_]))
        # print(len(hash_value))
    except IndexError:
        print("image too small, skipped")
        return 0
    return hash_value, normalized_coeff


def compute_image_distances(msa_file):
    if msa_file == "neotrop_reference.fasta":
        query_file = msa_file.replace("_reference.fasta", "_query_10k.fasta")
    else:
        query_file = os.path.join(os.pardir,
                                  "data/processed/loo/merged_" + msa_file.replace("_reference.fasta", ".fasta"))
    results = []
    counter = 0
    print(msa_file)
    isAA = False
    datatype = loo_selection[loo_selection["verbose_name"] == msa_file.replace("_reference.fasta", ".phy")].iloc[0][
        "data_type"]
    if datatype == "AA" or datatype == "DataType.AA":
        isAA = True
    print(isAA)  # Skip already processed
    print(msa_file)
    # Skip already processed
    potential_path = os.path.join(os.pardir, "data/processed/features",
                                  msa_file.replace("_reference.fasta", "") + "_msa_im_comp" + ".csv")
    if os.path.exists(potential_path):
        print("Skipped Image Comp: " + msa_file + " already processed")
        # return 0

    # current_loo_targets = pd.read_csv(os.path.join(os.pardir, "data/processed/target", "loo_result_entropy.csv"))
    # sampledData = current_loo_targets[current_loo_targets["dataset"] == msa_file.replace("_reference.fasta", "")][
    #   "sampleId"].values.tolist()

    for record_query in SeqIO.parse(os.path.join(os.pardir, "data/raw/query", query_file), 'fasta'):
        counter += 1
        if counter % 50 == 0:
            print(counter)
        # if record_query.name not in sampledData:
        #   continue
        distances_hu = []
        distances_lbp = []
        distances_pca = []

        numeric_query = dna_to_numeric(record_query.seq, isAA)
        image_query = encode_dna_as_image(numeric_query)
        image_query = image_query.astype(np.uint8)
        image_query_hu = image_query
        image_query_hu[image_query_hu != 0] = 1
        lbp_hist_query = lbp_histogram(image_query)

        for record_msa in SeqIO.parse(os.path.join(os.pardir, "data/raw/msa", msa_file), 'fasta'):
            if record_msa.id != record_query.id:
                numeric_req = dna_to_numeric(record_msa.seq, isAA)
                image_msa_req = encode_dna_as_image(numeric_req)
                image_msa_req = image_msa_req.astype(np.uint8)

                image_msa_req_hu = image_msa_req
                image_msa_req_hu[image_msa_req_hu != 0] = 1

                contours, hierarchy = cv2.findContours(image_query_hu, 2, 1)
                cnt1 = contours[0]
                contours, hierarchy = cv2.findContours(image_msa_req_hu, 2, 1)
                cnt2 = contours[0]

                moments1 = cv2.moments(cnt1)
                hu_moments1 = cv2.HuMoments(moments1)

                moments2 = cv2.moments(cnt2)
                hu_moments2 = cv2.HuMoments(moments2)

                distances_hu.append(np.sqrt(np.sum((hu_moments1 - hu_moments2) ** 2))
                                    )

                lbp_msa_req = lbp_histogram(image_msa_req)
                lbp_dist = euclidean(lbp_msa_req, lbp_hist_query)
                distances_lbp.append(lbp_dist)

                num_components = 10
                try:
                    pca1 = PCA(n_components=num_components)
                    pca2 = PCA(n_components=num_components)
                    pca_components1 = pca1.fit_transform(normalize(image_query, axis=1, norm="l1"))
                    pca_components2 = pca2.fit_transform(normalize(image_msa_req, axis=1, norm="l1"))
                    distance_pca = np.linalg.norm(pca_components1 - pca_components2)

                    distances_pca.append(distance_pca)
                except (ValueError, np.linalg.LinAlgError):
                    try:
                        print("Value error occured")
                        pca1 = PCA(n_components=5)
                        pca2 = PCA(n_components=5)
                        pca_components1 = pca1.fit_transform(normalize(image_query, axis=1, norm="l1"))
                        pca_components2 = pca2.fit_transform(normalize(image_msa_req, axis=1, norm="l1"))
                        distance_pca = np.linalg.norm(pca_components1 - pca_components2)

                        distances_pca.append(distance_pca)
                    except (ValueError, np.linalg.LinAlgError):
                        try:
                            print("Value error occured")
                            pca1 = PCA(n_components=1)
                            pca2 = PCA(n_components=1)
                            pca_components1 = pca1.fit_transform(normalize(image_query, axis=1, norm="l1"))
                            pca_components2 = pca2.fit_transform(normalize(image_msa_req, axis=1, norm="l1"))
                            distance_pca = np.linalg.norm(pca_components1 - pca_components2)
                            print(distances_pca)
                            distances_pca.append(distance_pca)
                        except (ValueError, np.linalg.LinAlgError):
                            print("Skipped, error occured in PCA")
        try:
            min_distance = min(distances_hu)
            max_distance = max(distances_hu)
        except ValueError:
            return 0

        max_dist_hu = max(distances_hu)
        min_dist_hu = min(distances_hu)
        avg_dist_hu = sum(distances_hu) / len(distances_hu)
        std_dist_hu = statistics.stdev(distances_hu)

        sk_dist_hu = skew(distances_hu)
        kur_dist_hu = kurtosis(distances_hu, fisher=False)

        # max_dist_hu = max_dist_hu / len(distances_hu)
        #  min_dist_hu = min_dist_hu / len(distances_hu)
        # avg_dist_hu = avg_dist_hu / len(distances_hu)
        # std_dist_hu = std_dist_hu / len(distances_hu)

        max_dist_lbp = max(distances_lbp)
        min_dist_lbp = min(distances_lbp)
        avg_dist_lbp = sum(distances_lbp) / len(distances_lbp)
        std_dist_lbp = statistics.stdev(distances_lbp)

        sk_dist_lbp = skew(distances_lbp)
        kur_dist_lbp = kurtosis(distances_lbp, fisher=False)

        max_dist_lbp = max_dist_lbp / len(distances_lbp)
        min_dist_lbp = min_dist_lbp / len(distances_lbp)
        avg_dist_lbp = avg_dist_lbp / len(distances_lbp)
        std_dist_lbp = std_dist_lbp / len(distances_lbp)

        max_dist_pca = max(distances_pca)
        min_dist_pca = min(distances_pca)
        avg_dist_pca = sum(distances_pca) / len(distances_pca)
        std_dist_pca = statistics.stdev(distances_pca)

        sk_dist_pca = skew(distances_pca)
        kur_dist_pca = kurtosis(distances_pca, fisher=False)

        max_dist_pca = max_dist_pca / len(distances_pca)
        min_dist_pca = min_dist_pca / len(distances_pca)
        avg_dist_pca = avg_dist_pca / len(distances_pca)
        std_dist_pca = std_dist_pca / len(distances_pca)

        name = ""

        if msa_file == "neotrop_reference.fasta":
            name = "neotrop"
        elif msa_file == "bv_reference.fasta":
            name = "bv"
        elif msa_file == "tara_reference.fasta":
            name = "tara"
        else:
            name = msa_file.replace("_reference.fasta", "")

        results.append(
            (name, record_query.id, max_dist_hu, min_dist_hu, avg_dist_hu, std_dist_hu, sk_dist_hu, kur_dist_hu,
             sk_dist_lbp, kur_dist_lbp, max_dist_lbp, min_dist_lbp, avg_dist_lbp, std_dist_lbp, sk_dist_pca,
             kur_dist_pca,
             max_dist_pca, min_dist_pca, avg_dist_pca, std_dist_pca))
    return results, msa_file


def compute_perceptual_hash_distance(msa_file):
    if msa_file == "neotrop_reference.fasta":
        query_file = msa_file.replace("_reference.fasta", "_query_10k.fasta")
    else:
        query_file = msa_file.replace("_reference.fasta", "_query.fasta")
    results = []
    counter = 0
    print(msa_file)
    isAA = False
    datatype = loo_selection[loo_selection["verbose_name"] == msa_file.replace("_reference.fasta", ".phy")].iloc[0][
        "data_type"]
    if datatype == "AA" or datatype == "DataType.AA":
        isAA = True
    print(isAA)  # Skip already processed
    potential_path = os.path.join(os.pardir, "data/processed/features",
                                  msa_file.replace("_reference.fasta", "") + "16p_msa_perc_hash_dist" + ".csv")
    if os.path.exists(potential_path):
        print("Skipped: " + msa_file + " already processed")

    current_loo_targets = pd.read_csv(os.path.join(os.pardir, "data/processed/target", "loo_result_entropy.csv"))
    sampledData = current_loo_targets[current_loo_targets["dataset"] == msa_file.replace("_reference.fasta", "")][
        "sampleId"].values.tolist()

    for record_query in SeqIO.parse(os.path.join(os.pardir, "data/raw/query", query_file), 'fasta'):

        counter += 1
        if counter % 50 == 0:
            print(counter)
        # if record_query.name not in sampledData:
        #   continue
        distances = []
        kmer_sims10 = []
        kmer_sims15 = []
        kmer_sims25 = []
        kmer_sims50 = []
        lcs_values = []
        coeff_dists = []
        hash_query, normalized_query_dct_coeff = compute_dct_sign_only_hash(record_query.seq, isAA)
        current_closest_taxon = ""
        current_min_distance = 10000000000000000000000000
        for record_msa in SeqIO.parse(os.path.join(os.pardir, "data/raw/msa", msa_file), 'fasta'):
            if record_msa.id != record_query.id:
                hash_msa, normalized_msa_dct_coeff = compute_dct_sign_only_hash(record_msa.seq, isAA)
                if hash_msa != 0:
                    distance = compute_hamming_distance(hash_msa, hash_query)
                    if current_min_distance == 10000000000000000000000000:
                        current_min_distance = distance
                        current_closest_taxon = record_msa.id
                    if distance < current_min_distance:
                        current_min_distance = distance
                        current_closest_taxon = record_msa.id
                    lcs = pylcs.lcs_sequence_length(hash_msa, hash_query)
                    distances.append(distance)
                    kmer_sim10 = fraction_shared_kmers(hash_msa, hash_query, 10)
                    kmer_sim15 = fraction_shared_kmers(hash_msa, hash_query, 15)
                    kmer_sim25 = fraction_shared_kmers(hash_msa, hash_query, 25)
                    kmer_sim50 = fraction_shared_kmers(hash_msa, hash_query, 50)
                    kmer_sims15.append(kmer_sim15)
                    kmer_sims10.append(kmer_sim10)
                    kmer_sims25.append(kmer_sim25)
                    kmer_sims50.append(kmer_sim50)
                    lcs_values.append(lcs)
                    difference = normalized_msa_dct_coeff.flatten() - normalized_query_dct_coeff.flatten()
                    coeff_dist = np.linalg.norm(difference)
                    coeff_dists.append(coeff_dist)
                else:
                    return 0
        if len(distances) < 2:
            return 0

        max_ham = max(distances)
        min_ham = min(distances)
        avg_ham = sum(distances) / len(distances)
        std_ham = statistics.stdev(distances)

        sk_ham = skew(distances)
        kur_ham = kurtosis(distances, fisher=True)

        rel_max_ham = max_ham / len(hash_query)
        rel_min_ham = min_ham / len(hash_query)
        rel_avg_ham = avg_ham / len(hash_query)
        rel_std_ham = std_ham / len(hash_query)

        max_dist_coeff = max(coeff_dists)
        min_dist_coeff = min(coeff_dists)
        avg_dist_coeff = sum(coeff_dists) / len(coeff_dists)
        std_dist_coeff = statistics.stdev(coeff_dists)

        sk_dist_coeff = skew(coeff_dists)
        kur_dist_coeff = kurtosis(coeff_dists, fisher=True)

        max_kmer_sim10 = max(kmer_sims10)
        min_kmer_sim10 = min(kmer_sims10)
        avg_kmer_sim10 = sum(kmer_sims10) / len(kmer_sims10)
        std_kmer_sim10 = statistics.stdev(kmer_sims10)

        sk_kmer_sim10 = skew(kmer_sims10)
        kur_kmer_sim10 = kurtosis(kmer_sims10, fisher=True)

        rel_max_kmer_sim10 = max_kmer_sim10
        rel_min_kmer_sim10 = min_kmer_sim10
        rel_avg_kmer_sim10 = avg_kmer_sim10
        rel_std_kmer_sim10 = std_kmer_sim10

        max_kmer_sim15 = max(kmer_sims15)
        min_kmer_sim15 = min(kmer_sims15)
        avg_kmer_sim15 = sum(kmer_sims15) / len(kmer_sims15)
        std_kmer_sim15 = statistics.stdev(kmer_sims15)

        sk_kmer_sim15 = skew(kmer_sims15)
        kur_kmer_sim15 = kurtosis(kmer_sims15, fisher=True)

        rel_max_kmer_sim15 = max_kmer_sim15
        rel_min_kmer_sim15 = min_kmer_sim15
        rel_avg_kmer_sim15 = avg_kmer_sim15
        rel_std_kmer_sim15 = std_kmer_sim15

        max_kmer_sim25 = max(kmer_sims25)
        min_kmer_sim25 = min(kmer_sims25)
        avg_kmer_sim25 = sum(kmer_sims25) / len(kmer_sims25)
        std_kmer_sim25 = statistics.stdev(kmer_sims25)

        sk_kmer_sim25 = skew(kmer_sims25)
        kur_kmer_sim25 = kurtosis(kmer_sims25, fisher=True)

        rel_max_kmer_sim25 = max_kmer_sim25
        rel_min_kmer_sim25 = min_kmer_sim25
        rel_avg_kmer_sim25 = avg_kmer_sim25
        rel_std_kmer_sim25 = std_kmer_sim25

        max_kmer_sim50 = max(kmer_sims50)
        min_kmer_sim50 = min(kmer_sims50)
        avg_kmer_sim50 = sum(kmer_sims50) / len(kmer_sims50)
        std_kmer_sim50 = statistics.stdev(kmer_sims50)

        sk_kmer_sim50 = skew(kmer_sims50)
        kur_kmer_sim50 = kurtosis(kmer_sims50, fisher=True)

        rel_max_kmer_sim50 = max_kmer_sim50
        rel_min_kmer_sim50 = min_kmer_sim50
        rel_avg_kmer_sim50 = avg_kmer_sim50
        rel_std_kmer_sim50 = std_kmer_sim50

        max_ham_lcs = max(lcs_values)
        min_ham_lcs = min(lcs_values)
        avg_ham_lcs = sum(lcs_values) / len(lcs_values)
        std_ham_lcs = statistics.stdev(lcs_values)

        sk_ham_lcs = skew(lcs_values)
        kur_ham_lcs = kurtosis(lcs_values, fisher=True)

        rel_max_ham_lcs = max_ham_lcs / len(hash_query)
        rel_min_ham_lcs = min_ham_lcs / len(hash_query)
        rel_avg_ham_lcs = avg_ham_lcs / len(hash_query)
        rel_std_ham_lcs = std_ham_lcs / len(hash_query)

        name = ""

        if msa_file == "neotrop_reference.fasta":
            name = "neotrop"
        elif msa_file == "bv_reference.fasta":
            name = "bv"
        elif msa_file == "tara_reference.fasta":
            name = "tara"
        else:
            name = msa_file.replace("_reference.fasta", "")

        results.append((
                       name, record_query.id, current_closest_taxon, rel_min_ham, rel_max_ham, rel_avg_ham, rel_std_ham,
                       sk_ham, kur_ham,
                       sk_kmer_sim10, kur_kmer_sim10, rel_max_kmer_sim10, rel_min_kmer_sim10, rel_avg_kmer_sim10,
                       rel_std_kmer_sim10,
                       sk_kmer_sim15, kur_kmer_sim15, rel_max_kmer_sim15, rel_min_kmer_sim15, rel_avg_kmer_sim15,
                       rel_std_kmer_sim15,
                       sk_kmer_sim25, kur_kmer_sim25, rel_max_kmer_sim25, rel_min_kmer_sim25, rel_avg_kmer_sim25,
                       rel_std_kmer_sim25,
                       sk_kmer_sim50, kur_kmer_sim50, rel_max_kmer_sim50, rel_min_kmer_sim50, rel_avg_kmer_sim50,
                       rel_std_kmer_sim50,
                       sk_ham_lcs,
                       kur_ham_lcs, rel_max_ham_lcs, rel_min_ham_lcs, rel_avg_ham_lcs, rel_std_ham_lcs,
                       max_dist_coeff, min_dist_coeff, std_dist_coeff, avg_dist_coeff, sk_dist_coeff, kur_dist_coeff))
    return results, msa_file


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

    for file in filenames:
        print(file)
        if not os.path.exists(os.path.join(os.pardir, "data/raw/msa", file)):
            print("File not found: " + file)
            filenames.remove(file)
            continue
        if file == "11274_2_reference.fasta":
            filenames.remove(file)

        if len(next(SeqIO.parse(os.path.join(os.pardir, "data/raw/msa", file), 'fasta').records).seq) > 15000:
            filenames.remove(file)
    filenames_comp = filenames

    if multiprocessing.current_process().name == 'MainProcess':
        multiprocessing.freeze_support()

    pool = multiprocessing.Pool()
    results = pool.imap_unordered(compute_perceptual_hash_distance, filenames)

    for result in results:
       if result != 0:
        print("Finished processing: " + result[1] + "with query file")
        df = pd.DataFrame(result[0],
                          columns=['dataset', 'sampleId', 'current_closest_taxon_perc_ham','min_perc_hash_ham_dist', 'max_perc_hash_ham_dist',
                                  'avg_perc_hash_ham_dist',
                                 'std_perc_hash_ham_dist', 'sks_perc_hash_ham_dist',
                                'kur_perc_hash_ham_dist', "sk_kmer_sim10", "kur_kmer_sim10",
                               "rel_max_kmer_sim10", "rel_min_kmer_sim10", "rel_avg_kmer_sim10",
                              "rel_std_kmer_sim10",
                             "sk_kmer_sim15", "kur_kmer_sim15", "rel_max_kmer_sim15", "rel_min_kmer_sim15",
                            "rel_avg_kmer_sim15",
                           "rel_std_kmer_sim15",
                          "sk_kmer_sim25", "kur_kmer_sim25", "rel_max_kmer_sim25", "rel_min_kmer_sim25",
                         "rel_avg_kmer_sim25",
                        "rel_std_kmer_sim25",
                       "sk_kmer_sim50", "kur_kmer_sim50", "rel_max_kmer_sim50", "rel_min_kmer_sim50",
                      "rel_avg_kmer_sim50",
                     "rel_std_kmer_sim50",
                    "sk_perc_hash_lcs",
                   "kur_perc_hash_lcs", "max_perc_hash_lcs", "min_perc_hash_lcs",
                  "avg_perc_hash_lcs",
                 "std_perc_hash_lcs",
                "max_dist_coeff", "min_dist_coeff", "std_dist_coeff", "avg_dist_coeff", "sk_dist_coeff",
           "kur_dist_coeff"
        ])
        df.to_csv(os.path.join(os.pardir, "data/processed/features",
                           result[1].replace("_reference.fasta", "") + str(
                              feature_config.SIGN_ONLY_MATRIX_SIZE) + "p_200_msa_perc_hash_dist.csv"))

    pool.close()
    pool.join()

    print("Finished Perc Hash, starting CV")
    # for file in filenames_comp:
    # if os.path.exists(os.path.join(os.pardir, "data/processed/features",
    #                          file.replace("_reference.fasta", "") + "_msa_im_comp.csv")):
    #  print("Found existing one ... ")
    # filenames_comp.remove(file)
    print(len(filenames_comp))
    #pool = multiprocessing.Pool(multiprocessing.cpu_count())
    #results = pool.imap_unordered(compute_image_distances, filenames_comp)

    #for result in results:
     #   if result != 0:
      #      print("Finished processing: " + result[1] + "with query file CV")
       #     df = pd.DataFrame(result[0],
        #                      columns=['dataset', 'sampleId', "max_dist_hu", "min_dist_hu", "avg_dist_hu",
         #                              "std_dist_hu", "sk_dist_hu", "kur_dist_hu",
          #                             "sk_dist_lbp", "kur_dist_lbp", "max_dist_lbp", "min_dist_lbp", "avg_dist_lbp",
           #                            "std_dist_lbp", "sk_dist_pca", "kur_dist_pca",
            #                           "max_dist_pca", "min_dist_pca", "avg_dist_pca", "std_dist_pca"])
            #df.to_csv(os.path.join(os.pardir, "data/processed/features",
             #                      result[1].replace("_reference.fasta", "") + "_msa_im_comp.csv"))

    #pool.close()
    #pool.join()
