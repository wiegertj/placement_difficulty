import multiprocessing
import statistics
import sys
import types
import numpy as np
import pylcs

import pandas as pd
import os
from Bio import SeqIO
from scipy.fftpack import dct
from collections import defaultdict
from scipy.stats import kurtosis, skew
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from skimage import measure
import cv2
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import normalize


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


def dna_to_numeric(sequence):
    mapping = {'A': 63, 'C': 127, 'G': 191, 'T': 255, '-': 0, 'N': 0}
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


def compute_dct_sign_only_hash(sequence):
    numeric_sequence = dna_to_numeric(sequence)
    image = encode_dna_as_image(numeric_sequence)

    dct_coeffs = dct(dct(image, axis=0), axis=1)
    sign_only_sequence = np.sign(dct_coeffs)
    size_ = 16

    try:
        sign_only_sequence = sign_only_sequence[np.ix_(list(range(size_)),
                                                       list(range(size_)))]
        hash_value = "".join([str(int(sign)) for sign in sign_only_sequence.flatten()])
    except IndexError:
        print("image too small, skipped")
        return 0
    return hash_value


def compute_image_distances(msa_file):
    if msa_file == "neotrop_reference.fasta":
        query_file = msa_file.replace("_reference.fasta", "_query_10k.fasta")
    else:
        query_file = msa_file.replace("_reference.fasta", "_query.fasta")
    results = []
    counter = 0
    print(msa_file)
    # Skip already processed
    potential_path = os.path.join(os.pardir, "data/processed/features",
                                  msa_file.replace("_reference.fasta", "") + "_msa_im_comp" + ".csv")
    if os.path.exists(potential_path):
        print("Skipped Image Comp: " + msa_file + " already processed")
        return 0

    for record_query in SeqIO.parse(os.path.join(os.pardir, "data/raw/query", query_file), 'fasta'):
        counter += 1
        if counter % 50 == 0:
            print(counter)
        distances_hu = []
        distances_lbp = []
        distances_pca = []

        numeric_query = dna_to_numeric(record_query.seq)
        image_query = encode_dna_as_image(numeric_query)
        image_query = image_query.astype(np.uint8)
        image_query_hu = image_query
        image_query_hu[image_query_hu != 0] = 1
        contours_query = cv2.findContours(image_query, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # contours1 = measure.find_contours(image_query, 0.5)
        # print(contours)
        # hu_moments1 = calculate_hu_moments(contours[0])
        lbp_hist_query = lbp_histogram(image_query)

        for record_msa in SeqIO.parse(os.path.join(os.pardir, "data/raw/msa", msa_file), 'fasta'):
            if record_msa.id != record_query.id:
                numeric_req = dna_to_numeric(record_msa.seq)
                image_msa_req = encode_dna_as_image(numeric_req)
                image_msa_req = image_msa_req.astype(np.uint8)
                # print(image_msa_req)

                # ret, thresh = cv2.threshold(image_msa_req, 1, 255, 0)
                # ret, thresh2 = cv2.threshold(image_query, 1, 255, 0)
                image_msa_req_hu = image_msa_req
                image_msa_req_hu[image_msa_req_hu != 0] = 1

                contours, hierarchy = cv2.findContours(image_query_hu, 2, 1)
                cnt1 = contours[0]
                contours, hierarchy = cv2.findContours(image_msa_req_hu, 2, 1)
                cnt2 = contours[0]

                # Compute Hu moments for the first contour
                moments1 = cv2.moments(cnt1)
                hu_moments1 = cv2.HuMoments(moments1)

                # Compute Hu moments for the second contour
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
                except ValueError:
                    try:
                        print("Value error occured")
                        pca1 = PCA(n_components=5)
                        pca2 = PCA(n_components=5)
                        pca_components1 = pca1.fit_transform(normalize(image_query, axis=1, norm="l1"))
                        pca_components2 = pca2.fit_transform(normalize(image_msa_req, axis=1, norm="l1"))
                        distance_pca = np.linalg.norm(pca_components1 - pca_components2)

                        distances_pca.append(distance_pca)
                    except ValueError:
                        print("Value error occured")
                        pca1 = PCA(n_components=1)
                        pca2 = PCA(n_components=1)
                        pca_components1 = pca1.fit_transform(normalize(image_query, axis=1, norm="l1"))
                        pca_components2 = pca2.fit_transform(normalize(image_msa_req, axis=1, norm="l1"))
                        distance_pca = np.linalg.norm(pca_components1 - pca_components2)
                        print(distances_pca)
                        distances_pca.append(distance_pca)

        min_distance = min(distances_hu)
        max_distance = max(distances_hu)

        # Normalize distances using Min-Max scaling
        # print(distances_hu)
        # distances_hu = [(d - min_distance) / (max_distance - min_distance) for d in distances_hu]

        # print(distances_hu)

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
    # Skip already processed
    potential_path = os.path.join(os.pardir, "data/processed/features",
                                  msa_file.replace("_reference.fasta", "") + "16p_msa_perc_hash_dist" + ".csv")
    if os.path.exists(potential_path):
        print("Skipped: " + msa_file + " already processed")
        return 0

    for record_query in SeqIO.parse(os.path.join(os.pardir, "data/raw/query", query_file), 'fasta'):
        counter += 1
        if counter % 50 == 0:
            print(counter)
        distances = []
        distances_cosine = []
        lcs_values = []
        hash_query = compute_dct_sign_only_hash(record_query.seq)

        for record_msa in SeqIO.parse(os.path.join(os.pardir, "data/raw/msa", msa_file), 'fasta'):
            if record_msa.id != record_query.id:
                hash_msa = compute_dct_sign_only_hash(record_msa.seq)
                if hash_msa != 0:
                    distance = compute_hamming_distance(hash_msa, hash_query)
                    print(hash_query)
                    distance_cosine = cosine_similarity(hash_msa.replace("-", 0), hash_query.replace("-",0))
                    lcs = pylcs.lcs_sequence_length(hash_msa, hash_query)
                    #print(lcs)
                    distances.append(distance)
                    distances_cosine.append(distance_cosine)
                    lcs_values.append(lcs)
                else:
                    return 0

        max_ham = max(distances)
        min_ham = min(distances)
        avg_ham = sum(distances) / len(distances)
        std_ham = statistics.stdev(distances)

        sk_ham = skew(distances)
        kur_ham = kurtosis(distances, fisher=False)

        rel_max_ham = max_ham / len(hash_query)
        rel_min_ham = min_ham / len(hash_query)
        rel_avg_ham = avg_ham / len(hash_query)
        rel_std_ham = std_ham / len(hash_query)

        max_ham_cos = max(distances_cosine)
        min_ham_cos = min(distances_cosine)
        avg_ham_cos = sum(distances_cosine) / len(distances_cosine)
        std_ham_cos = statistics.stdev(distances_cosine)

        sk_ham_cos = skew(distances_cosine)
        kur_ham_cos = kurtosis(distances_cosine, fisher=False)

        rel_max_ham_cos = max_ham_cos / len(distances_cosine)
        rel_min_ham_cos = min_ham_cos / len(distances_cosine)
        rel_avg_ham_cos = avg_ham_cos / len(distances_cosine)
        rel_std_ham_cos = std_ham_cos / len(distances_cosine)

        max_ham_lcs = max(lcs_values)
        min_ham_lcs = min(lcs_values)
        avg_ham_lcs = sum(lcs_values) / len(lcs_values)
        std_ham_lcs = statistics.stdev(lcs_values)

        sk_ham_lcs = skew(lcs_values)
        kur_ham_lcs = kurtosis(lcs_values, fisher=False)

        rel_max_ham_lcs = max_ham_lcs / len(lcs_values)
        rel_min_ham_lcs = min_ham_lcs / len(lcs_values)
        rel_avg_ham_lcs = avg_ham_lcs / len(lcs_values)
        rel_std_ham_lcs = std_ham_lcs / len(lcs_values)

        name = ""

        if msa_file == "neotrop_reference.fasta":
            name = "neotrop"
        elif msa_file == "bv_reference.fasta":
            name = "bv"
        elif msa_file == "tara_reference.fasta":
            name = "tara"
        else:
            name = msa_file.replace("_reference.fasta", "")

        results.append((name, record_query.id, rel_min_ham, rel_max_ham, rel_avg_ham, rel_std_ham, sk_ham, kur_ham,
                        sk_ham_cos, kur_ham_cos, rel_max_ham_cos, rel_min_ham_cos, rel_avg_ham_cos, rel_std_ham_cos,
                        sk_ham_lcs,
                        kur_ham_lcs, rel_max_ham_lcs, rel_min_ham_lcs, rel_avg_ham_lcs, rel_std_ham_lcs))
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
        if not os.path.exists(os.path.join(os.pardir, "data/raw/msa", file)):
            print("File not found: " + file)
            filenames.remove(file)
            continue

        # if len(next(SeqIO.parse(os.path.join(os.pardir, "data/raw/msa", file), 'fasta').records).seq) > 15000:
        #   filenames.remove(file)
    filenames_comp = filenames

    if multiprocessing.current_process().name == 'MainProcess':
        multiprocessing.freeze_support()

    pool = multiprocessing.Pool()
    results = pool.imap_unordered(compute_perceptual_hash_distance, filenames)

    for result in results:
        if result != 0:
            print("Finished processing: " + result[1] + "with query file")
            df = pd.DataFrame(result[0],
                              columns=['dataset', 'sampleId', 'min_perc_hash_ham_dist', 'max_perc_hash_ham_dist',
                                       'avg_perc_hash_ham_dist',
                                       'std_perc_hash_ham_dist', 'skewness_perc_hash_ham_dist',
                                       'kurtosis_perc_hash_ham_dist', "sk_ham_cos", "kur_ham_cos", "rel_max_ham_cos",
                                       "rel_min_ham_cos", "rel_avg_ham_cos", "rel_std_ham_cos", "sk_ham_lcs",
                                       "kur_ham_lcs", "rel_max_ham_lcs", "rel_min_ham_lcs", "rel_avg_ham_lcs",
                                       "rel_std_ham_lcs"
                                       ])
            df.to_csv(os.path.join(os.pardir, "data/processed/features",
                                   result[1].replace("_reference.fasta", "") + str(
                                       feature_config.SIGN_ONLY_MATRIX_SIZE) + "p_msa_perc_hash_dist.csv"))

    pool.close()
    pool.join()

    sys.exit()

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = pool.imap_unordered(compute_image_distances, filenames_comp)

    for result in results:
        if result != 0:
            print("Finished processing: " + result[1] + "with query file")
            df = pd.DataFrame(result[0],
                              columns=['dataset', 'sampleId', "max_dist_hu", "min_dist_hu", "avg_dist_hu",
                                       "std_dist_hu", "sk_dist_hu", "kur_dist_hu",
                                       "sk_dist_lbp", "kur_dist_lbp", "max_dist_lbp", "min_dist_lbp", "avg_dist_lbp",
                                       "std_dist_lbp", "sk_dist_pca", "kur_dist_pca",
                                       "max_dist_pca", "min_dist_pca", "avg_dist_pca", "std_dist_pca"])
            df.to_csv(os.path.join(os.pardir, "data/processed/features",
                                   result[1].replace("_reference.fasta", "") + "_msa_im_comp.csv"))

    pool.close()
    pool.join()
