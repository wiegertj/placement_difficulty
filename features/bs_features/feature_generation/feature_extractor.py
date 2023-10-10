import logging
import pandas as pd
import numpy as np
import os

from Bio import Phylo


class FeatureExtractor:
    def __init__(self, msa_file_path, tree_file_path, model_file_path, feature_computer):
        self.msa_file_path = msa_file_path
        self.tree_file_path = tree_file_path
        self.model_file_path = model_file_path
        self.feature_computer = feature_computer

        # Initialize a logger
        self.logger = self.setup_logger()

        # Check if the specified files exist
        self.check_file_exists(msa_file_path, "MSA file")
        self.check_file_exists(tree_file_path, "tree file")
        self.check_file_exists(model_file_path, "model file")

    def setup_logger(self):
        logger = logging.getLogger('FeatureExtractor')
        logger.setLevel(logging.DEBUG)

        # Create a console handler and set the logging level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create a formatter and add it to the console handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        # Add the console handler to the logger
        logger.addHandler(ch)

        return logger

    def check_file_exists(self, file_path, file_description):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"{file_description} not found: {file_path}")

    def load_msa(self):
        self.logger.info("Loading MSA from: %s", self.msa_file_path)
        # Load the MSA from the specified file path (e.g., FASTA format)
        # Return the MSA data in a suitable data structure (e.g., a DataFrame)
        msa_data = pd.read_fasta(self.msa_file_path)
        return msa_data

    def load_tree(self):
        self.logger.info("Loading tree from: %s", self.tree_file_path)
        # Load the phylogenetic tree from the specified tree file path (e.g., Newick format)
        # Return the tree data in a suitable data structure (e.g., a tree object)
        tree_data = Phylo.read(self.tree_file_path, 'newick')
        return tree_data

    def load_model(self):
        self.logger.info("Loading model from: %s", self.model_file_path)
        # Load the model from the specified model file path (e.g., using a machine learning library)
        # Return the trained model
        model = load_model(self.model_file_path)
        return model

    def extract_features(self):
        # Load the MSA, tree, and model
        msa_data = self.load_msa()
        tree_data = self.load_tree()
        model = self.load_model()

        self.logger.info("Performing feature extraction...")



        # Return the extracted features as a feature matrix (e.g., a NumPy array)
        feature_matrix = np.zeros((len(msa_data), 4))  # Adjust num_features as needed
        return feature_matrix


if __name__ == "__main__":
    # Example usage
    msa_file = "your_msa.fasta"
    tree_file = "your_tree.newick"
    model_file = "your_model.pkl"

    feature_extractor = FeatureExtractor(msa_file, tree_file, model_file)
    features = feature_extractor.extract_features()
