import statistics
import types
import ete3
import numpy as np
import os
import pandas as pd
from scipy.stats import skew, kurtosis
import dendropy
from sklearn.decomposition import PCA
import networkx as nx
from scipy.spatial.distance import pdist

import numpy as np
from scipy.stats import skew, kurtosis
from node2vec import Node2Vec

def normalize_branch_lengths(tree):
    total_length = 0.0

    for node in tree.traverse():
        if not node.is_root():
            total_length += node.dist

    for node in tree.traverse():
        if node.up:
            node.dist /= total_length
    return tree

def calc_tree_embedding(name, tree):

    # Create an empty directed graph
    G = nx.DiGraph()

    # Traverse the ETE3 tree and add edges to the graph with branch lengths as weights
    def traverse_and_add_edges(node):
        for child in node.children:
            edge_weight = node.get_distance(child)
            G.add_edge(node.name, child.name, weight=edge_weight)
            traverse_and_add_edges(child)

    # Start traversal from the tree root
    traverse_and_add_edges(tree)

    # Initialize and generate node embeddings using node2vec
    node2vec = Node2Vec(G, dimensions=5, walk_length=10, num_walks=100, workers=10)

    # Learn embeddings
    model = node2vec.fit(window=5, min_count=1)

    # Get the embeddings for nodes
    node_embeddings = {node: model.wv[node] for node in G.nodes()}
    print(node_embeddings)

    # Convert embeddings to a matrix
    embedding_matrix = np.array([node_embeddings[node] for node in G.nodes()])

    # Apply PCA to reduce dimensionality to 5 components
    pca = PCA(n_components=5)
    pca_result = pca.fit_transform(embedding_matrix)

    print("PCA Results (First Few Rows):")
    print(pca_result[:5])

    # Extract the embeddings as a NumPy array
    embeddings_array = np.array(list(node_embeddings.values()))

    pairwise_distances = pdist(embeddings_array, metric='euclidean')

    min_embedding = np.min(pairwise_distances)
    max_embedding = np.max(pairwise_distances)
    mean_embedding = np.mean(pairwise_distances)
    std_embedding = np.std(pairwise_distances)
    skewness_embedding = skew(pairwise_distances)
    kurtosis_embedding = kurtosis(pairwise_distances)

    return (name, min_embedding, max_embedding, mean_embedding, std_embedding, skewness_embedding, kurtosis_embedding)


if __name__ == '__main__':

    results = []
    grandir = os.path.join(os.getcwd(), os.pardir, os.pardir)

    module_path = os.path.join(grandir, "configs/feature_config.py")

    feature_config = types.ModuleType('feature_config')
    feature_config.__file__ = module_path

    with open(module_path, 'rb') as module_file:
        code = compile(module_file.read(), module_path, 'exec')
        exec(code, feature_config.__dict__)

    loo_selection = pd.read_csv(os.path.join(grandir, "data/loo_selection.csv"))
    filenames = loo_selection['verbose_name'].str.replace(".phy", ".newick").tolist()

    if feature_config.INCUDE_TARA_BV_NEO:
        filenames = filenames + ["bv_reference.fasta", "neotrop_reference.fasta", "tara_reference.fasta"]

    print(filenames)

    for file in filenames:
        if not os.path.exists(os.path.join(grandir, "data/raw/reference_tree", file)):
            print("Not found " + file)
            filenames.remove(file)
    counter = 0
    for tree_file in filenames:
        with open(os.path.join(grandir, "data/raw/reference_tree", tree_file), 'r') as file:
            newick_tree = file.read()

        counter += 1
        print(counter)

        tree = ete3.Tree(newick_tree)
        tree = normalize_branch_lengths(tree)

        embeds = calc_tree_embedding(tree_file.replace(".newick", ""), tree)
        embeds_list = [embeds]

        print("finished one embedding")

        df_tmp = pd.DataFrame(embeds_list, columns=["dataset","min_embedding", "max_embedding", "mean_embedding", "std_embedding", "skewness_embedding", "kurtosis_embedding"])


        if not os.path.isfile(os.path.join(grandir, "data/processed/features",
                                 "tree_embedd_stats.csv")):
            df_tmp.to_csv(os.path.join(os.path.join(grandir, "data/processed/features",
                                 "tree_embedd_stats.csv")), index=False)
        else:
            df_tmp.to_csv(os.path.join(grandir, "data/processed/features",
                                 "tree_embedd_stats.csv"),
                         index=False,
                         mode='a', header=False)
