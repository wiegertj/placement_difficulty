import os
import pandas as pd
import subprocess

filenames = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))["verbose_name"].str.replace(".phy", ".newick").values.tolist()
loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection_aa_test.csv"))
loo_selection["dataset"] = loo_selection["verbose_name"].str.replace(".phy", ".newick")
filenames_aa = loo_selection["dataset"].values.tolist()

duplicate_data = pd.read_csv(os.path.join(os.pardir, "data/treebase_difficulty_new.csv"))
accepted = []
counter = 0
filenames = filenames[:180]
filenames = filenames + filenames_aa
# Loop over each subdirectory (folder) within the specified path
counter = 0
results = []
for file in filenames:
    try:
        file = file.replace(".newick", "")
        file_path_parsimonies = f"/hits/fast/cme/wiegerjs/placement_difficulty/tests/{file}ebg_test/ebg_tmp/parsimony_bootstraps_tmp.txt"

        raxml_command = [
            "raxml-ng",
            "--consense MRE",
            f"--tree {file_path_parsimonies}",
            "--redo",
        ]

        subprocess.run(" ".join(raxml_command), shell=True)

        consensus_path = file_path_parsimonies + ".raxml.consensusTreeMRE"

        tree_path = os.path.join(os.pardir, "data/raw/reference_tree", file) + ".newick"

        print(tree_path)

        with open(consensus_path, 'r') as consensus_file, open(tree_path, 'r') as tree_file, open(file + "_combined_boot.txt",
                                                                                                  'w') as output_file:
            output_file.write(consensus_file.read())
            output_file.write(tree_file.read())
        combined_path = f"/hits/fast/cme/wiegerjs/placement_difficulty/scripts/{file}_combined_boot.txt"
        raxml_command = ["raxml-ng",
                         "--rfdist",
                         f"--tree {combined_path}",
                         "--redo"]
        # result =  subprocess.run(" ".join(raxml_command), shell=True)
        result = subprocess.run(" ".join(raxml_command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                shell=True)

        print("result")
        print(result.stdout)
        import re
        # Check if the command was successful
        if result.returncode == 0:
            # Extract numbers following "set: " using regular expressions
            numbers = re.findall(r'set:\s+(-?[\d.]+)', result.stdout)

            #  # Convert the extracted strings to integers or floats
            numbers = [int(num) if num.isdigit() else float(num) for num in numbers]

            #   # Print the extracted numbers
            print("Extracted numbers:", numbers)
        else:
            #   # Print an error message if the command failed
            print("Command failed with the following error message:")
            print(result.stderr)
        try:
            avg_rf_no_boot = numbers[0]
            avg_rel_rf_no_boot = numbers[1]
            no_top_no_boot = numbers[2]
        except (IndexError, NameError) as e:
            print("number extraction failed ....")
            continue
        results.append((avg_rel_rf_no_boot, file))
    except FileNotFoundError:
        continue
    ###########

print(results)
df = pd.DataFrame(results, columns=["rf_pars", "dataset"])
df.to_csv("rf_pars.csv", index=False)