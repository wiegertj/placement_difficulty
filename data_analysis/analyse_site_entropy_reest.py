import os
# Define the directory to search in
import subprocess
import dendropy
import ete3
import pandas as pd
from dendropy.calculate import treecompare


# Initialize a list to store the matching file paths
matching_files = []
directory_filtered = os.path.join(os.pardir, "data/raw/msa")
directory_reference = os.path.join(os.pardir, "data/raw/reference_tree")
# Walk through the directory and its subdirectories
results = []
for type in ["46", "37", "28", "19"]:
    for root, dirs, files in os.walk(directory_reference):
        for file in files:
            if file.endswith(".newick"):
                pot_path = directory_filtered + file.replace("tree", "msa").replace(".newick", "_filtered_" + type + ".raxml.bestTree")
                pot_path = pot_path.replace("msa", "msa/")
                print(pot_path)
                if os.path.exists(pot_path):
                    print("Found, started calculating")
                    start_index = directory_reference.replace("tree", "msa").replace(".newick", "_filtered_" + "" + ".bestTree").find("msa/") + len("msa/")
                    end_index = directory_reference.replace("tree", "msa").replace(".newick", "_filtered_" + "" + ".bestTree").find("_filtered")

                    # Extract the substring between the two substrings
                    desired_substring = pot_path.replace(".newick", "_filtered_" + "" + ".bestTree")[start_index:end_index]
                    print(desired_substring)
                    file_path = os.path.join(root, file)

                    with open(file_path, "r") as f:
                        original_newick_tree = f.read()
                    #original_newick_tree = file_path.read()
                    original_tree = ete3.Tree(original_newick_tree)
                    reest_tree = ete3.Tree(os.path.abspath(pot_path))

                    results_distance = original_tree.compare(reest_tree, unrooted=True)
                    print("Quartet paths")
                    print(os.path.abspath(pot_path))
                    print( os.path.abspath(file_path))
                    command = ["/home/wiegerjs/tqDist-1.0.2/bin/quartet_dist", "-v", os.path.abspath(pot_path),
                               os.path.abspath(file_path)]
                    try:
                        command_string = " ".join(command)
                        output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
                        lines = output.strip().split('\n')
                        values = lines[0].split()
                        quartet_distance = float(values[3])

                    except FileNotFoundError:
                        "Quartet File not found"

                    # BSD distance

                    tree_list = dendropy.TreeList()
                    tree_list.read(data=original_tree.write(format=1), schema="newick")
                    tree_list.read(data=reest_tree.write(format=1), schema="newick")

                    # Normalize Branch Lengths to be between 0 and 1
                    tree_len_1 = tree_list[0].length()
                    for edge in tree_list[0].postorder_edge_iter():
                        if edge.length is None:
                            edge.length = 0
                        else:
                            edge.length = float(edge.length) / tree_len_1

                    tree_len_2 = tree_list[1].length()
                    for edge in tree_list[1].postorder_edge_iter():
                        if edge.length is None:
                            edge.length = 0
                        else:
                            edge.length = float(edge.length) / tree_len_2

                    bsd_aligned = treecompare.euclidean_distance(tree_list[0], tree_list[1])

                    print(desired_substring)
                    print("Branch Score Distance (Aligned Trees):", bsd_aligned)
                    print("RF distance is %s over a total of" % (results_distance["norm_rf"]))
                    print("Quartet Distance: " + str(quartet_distance))

                    results.append((desired_substring, results_distance["norm_rf"], quartet_distance, bsd_aligned))

df = pd.DataFrame(results, columns=["dataset", "norm_rf", "norm_quart", "norm_bld"])
df.to_csv(os.path.join(os.pardir, "data/processed/final/site_ent_reest.csv"), index=False)
