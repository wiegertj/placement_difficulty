import os
import csv

path = "/Users/juliuswiegert/Downloads/Placements"
output_csv = "folder_names_raxml.csv"  # Name of the output CSV file

# List all items in the directory
items = os.listdir(path)

# Filter for subdirectories
subdirectories = [folder_name for folder_name in items if os.path.isdir(os.path.join(path, folder_name))]

# Write the list of subdirectories to a CSV file
with open(output_csv, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Folder Name'])  # Write a header row to the CSV file

    for folder_name in subdirectories:
        writer.writerow([folder_name])  # Write each folder name as a row in the CSV file
