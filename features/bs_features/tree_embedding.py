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
import numpy as np
from scipy.stats import skew, kurtosis
from node2vec import Node2Vec
def calc_tree_embedding(tree):

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
    node2vec = Node2Vec(G, dimensions=5, walk_length=10, num_walks=100, workers=4)

    # Learn embeddings
    model = node2vec.fit(window=5, min_count=1)

    # Get the embeddings for nodes
    node_embeddings = {node: model.wv[node] for node in G.nodes()}

    # Extract the embeddings as a NumPy array
    embeddings_array = np.array(list(node_embeddings.values()))

    # Compute the desired summary statistics
    min_embedding = np.min(embeddings_array, axis=0)
    max_embedding = np.max(embeddings_array, axis=0)
    mean_embedding = np.mean(embeddings_array, axis=0)
    std_embedding = np.std(embeddings_array, axis=0)
    skewness_embedding = skew(embeddings_array, axis=0)
    kurtosis_embedding = kurtosis(embeddings_array, axis=0)

    return (min_embedding, max_embedding, mean_embedding, std_embedding, skewness_embedding, kurtosis_embedding)


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

    for tree_file in filenames:
        with open(os.path.join(grandir, "data/raw/reference_tree", tree_file), 'r') as file:
            newick_tree = file.read()

    embeds = calc_tree_embedding(newick_tree)

    df_tmp = pd.DataFrame(embeds, columns=["min_embedding", "max_embedding", "mean_embedding", "std_embedding", "skewness_embedding", "kurtosis_embedding"])


    if not os.path.isfile(os.path.join(grandir, "data/processed/features",
                             "tree_embedd_stats.csv")):
        df_tmp.to_csv(os.path.join(os.path.join(grandir, "data/processed/features",
                             "tree_embedd_stats.csv")), index=False)
    else:
        df_tmp.to_csv(os.path.join(grandir, "data/processed/features",
                             "tree_embedd_stats.csv"),
                     index=False,
                     mode='a', header=False)
