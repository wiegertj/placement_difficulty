import os
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# Path to the CSV file
csv_file_path = '/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/features/bs_features/query200.csv'

# Read the CSV file
df = pd.read_csv(csv_file_path)
print(df.columns)

# Path to the directory containing the FASTA files
fasta_directory = '/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/loo'

# Initialize an empty list to store SeqRecord objects
for dataset in df['dataset'].unique():
    seq_records = []

    # Iterate over unique values in the "dataset" column
    # Get all values of "sampleId" for the current dataset
    sample_ids = df.loc[df['dataset'] == dataset, 'sampleIds']

    # Iterate over sampleIds and read the corresponding FASTA files
    for sample_id in sample_ids:
        # Construct the path to the FASTA file
        fasta_file_path = os.path.join(fasta_directory, f'{dataset}_query200_{sample_id}.fasta')

        # Check if the file exists before attempting to read it
        if os.path.exists(fasta_file_path):
            # Read the FASTA file
            records = SeqIO.parse(fasta_file_path, 'fasta')

            # Iterate over SeqRecord objects and append to the list
            for record in records:
                seq_records.append(SeqRecord(Seq(str(record.seq)), id=record.id, description=''))

# Path to the merged FASTA filef
    merged_fasta_file_path = os.path.join(fasta_directory, f'merged_{dataset}.fasta')

# Write the merged SeqRecord objects to a new FASTA file
    with open(merged_fasta_file_path, 'w') as output_file:
        SeqIO.write(seq_records, output_file, 'fasta')

    print(f'Merged FASTA file created: {merged_fasta_file_path}')
