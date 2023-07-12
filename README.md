# 1. Data Setup
---
### 1.1 For Standard Processing

Name of the dataset denoted as ...

- ./data/raw/msa contains reference MSA with name "..._reference.fasta" 

- ./data/raw/query contains query file with name "..._query.fasta"

- ./data/raw/placements contains placement files with name "..._epa_ result.jplace"

- ./data/raw/reference_tree contains reference tree with name "....newick"


### 1.2 For Leave One Out Processing
Each reference MSA sample will be once a query, the rest is used to reestimate the reference tree

Name of the dataset denoted as ...

- ./data/raw/msa contains reference MSA with name "..._reference.fasta" 

- ./data/raw/query contains query file with name "..._query.fasta"

- ./data/raw/reference_tree contains reference tree with name "....newick"

Query and reference are the same file in this case



# 2. Target Calculation
### 2.1 For Standard Processing

- ./target/placement_processing.py creates final target files in ./data/processed/target


### 2.2 For Leave One Out Processing


- ./scripts/leave_one_out.py** performs Leave One Out Processing using epa-ng and raxml-ng and returns placements in ./data/processed/loo_results

- ./target/placement_processing_loo.py creates final Leave One Out target file in ./data/processed/target


# 3. Feature Generation

## 3.1 k-mer Distance 

./features/kmer_processing.py: 