import os
import ete3
import subprocess
import glob
from Bio import AlignIO, SeqIO

for msa_name in ["21086_0"]:

    # Get all sequence Ids
    sequence_ids = []
    for record in SeqIO.parse(os.path.join(os.pardir, "data/raw/msa", msa_name + "_msa.fasta"), "fasta"):
        sequence_ids.append(record.id)

    filepath = os.path.join(os.pardir, "data/raw/msa", msa_name + "_msa.fasta")
    MSA = AlignIO.read(filepath, 'fasta')

    counter = 0

    # Perform LOO for each sequence
    for to_query in sequence_ids:
        counter += 1
        print(str(counter) + "/" + str(len(sequence_ids)))

        new_alignment = []
        query_alignment = []

        # Split MSA into query and rest
        for record in MSA:
            if record.id != to_query:
                seq_record = SeqIO.SeqRecord(seq=record.seq, id=record.id, description="")
                new_alignment.append(seq_record)
            else:
                seq_record = SeqIO.SeqRecord(seq=record.seq, id=record.id, description="")
                query_alignment.append(seq_record)

        # Write temporary query and MSA to files
        output_file = os.path.join(os.pardir, "data/processed/loo", msa_name + "_msa_" + to_query + ".fasta")
        output_file = os.path.abspath(output_file)

        output_file_query = os.path.join(os.pardir, "data/processed/loo", msa_name + "_query_" + to_query + ".fasta")
        output_file_query = os.path.abspath(output_file_query)

        SeqIO.write(new_alignment, output_file, "fasta")
        SeqIO.write(query_alignment, output_file_query, "fasta")

        # Get output tree path for result stroing
        output_file_tree = output_file.replace(".fasta", ".newick")

        # ------------------------------------------ run RAxML-ng with LOO MSA ------------------------------------------

        command = ["raxml-ng", "--search", "--msa", output_file, "--model",
                   os.path.join(os.pardir, "data/processed/loo", msa_name + "_msa_model.txt"), ]
        try:
            # Start the subprocess
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Wait for the process to complete and capture the output
            stdout, stderr = process.communicate()

            # Check the return code
            if process.returncode == 0:
                # Success
                print("RAxML-ng process completed successfully.")
                print("Output:")
                print(stdout)
            else:
                # Error
                print("RAxML-ng process failed with an error.")
                print("Error Output:")
                print(stderr)

        except FileNotFoundError:
            print("RAxML-ng executable not found. Please make sure RAxML-ng is installed and in the system")

        rel_tree_path = os.path.join(os.pardir, "data/processed/loo",
                                     msa_name + "_msa_" + to_query + ".fasta.raxml.bestTree")
        tree_path = os.path.abspath(rel_tree_path)
        with open(tree_path, 'r') as file:
            newick_tree = file.read()
            tree = ete3.Tree(newick_tree)

        # ------------------------------------ run epa-ng with new RAxML-ng tree ---------------------------------------

        # Create new directory for placement result
        os.mkdir(os.path.join(os.pardir, "data/processed/loo_results", msa_name + "_" + to_query))
        command = ["epa-ng", "--model", os.path.join(os.pardir, "data/processed/loo", msa_name + "_msa_model.txt"),
                   "--ref-msa", output_file, "--tree", tree_path, "--query", output_file_query, "--redo", "--outdir",
                   os.path.join(os.pardir, "data/processed/loo_results/" + msa_name + "_" + to_query), "--filter-max",
                   "10000", "--filter-acc-lwr", "0.99"]

        try:
            # Start the subprocess
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Wait for the process to complete and capture the output
            stdout, stderr = process.communicate()

            # Check the return code
            if process.returncode == 0:
                # Success
                print("EPA-ng process completed successfully.")
                print("Output:")
                print(stdout)
            else:
                # Error
                print("EPA-ng process failed with an error.")
                print("Error Output:")
                print(stderr)

        except FileNotFoundError:
            print("EPA-ng executable not found. Please make sure EPA-ng is installed and in the system PATH.")

        # ------------------------------------ Cleanup ---------------------------------------

        files = glob.glob(os.path.join(os.path.join(os.pardir, "data/processed/loo", f"*{to_query}*")))

        # Iterate over the temporary files and remove them
        for file_path in files:
            os.remove(file_path)
