import os
import subprocess

# Directories
raw_data_dir = "/hits/fast/cme/wiegerjs/EBG_train/EBG_train/data/raw"
bootstrap_dir = "/hits/fast/cme/wiegerjs/placement_difficulty/scripts"
raxml_executable = "/hits/fast/cme/wiegerjs/rapid_boot_ng_dev/raxml-ng-dev/build/bin/raxml-ng-adaptive"

# Iterate through each subfolder in the raw data directory
for root, dirs, files in os.walk(raw_data_dir):
    for subfolder in dirs:
        subfolder_path = os.path.join(root, subfolder)

        # Find the .newick file
        newick_path = None
        for file in os.listdir(subfolder_path):
            if file.endswith(".newick"):
                newick_path = os.path.join(subfolder_path, file)
                break

        # If no .newick file is found, skip this subfolder
        if not newick_path:
            print(f"No .newick file found in {subfolder_path}")
            continue

        # Check if the bootstrap file exists
        bootstrap_filepath = os.path.join(bootstrap_dir, f"{subfolder}_rb_ng.raxml.bootstraps")
        if os.path.exists(bootstrap_filepath):
            # Construct the raxml command
            output_prefix = f"{subfolder}_rapid_ng_support"
            raxml_command = [
                raxml_executable,
                "--support",
                f"--tree {newick_path}",
                f"--bs-trees {bootstrap_filepath}",
                "--redo",
                f"--prefix {output_prefix}"
            ]

            # Execute the command
            try:
                print(f"Running command for subfolder '{subfolder}': {' '.join(raxml_command)}")
                subprocess.run(raxml_command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while processing {subfolder}: {e}")
        else:
            print(f"Bootstrap file does not exist for {subfolder}: {bootstrap_filepath}")
