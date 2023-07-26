# -------------------- Feature config --------------------  #
K_MER_LENGTH = 15
K_MER_MAX_GAP_PERCENTAGE = 0.3
K_MER_MAX_N_PERCENTAGE = 0.3
BLOOM_FILTER_FP_RATE = 0.01
KMER_PROCESSING_INTERVAL_START = 0
KMER_PROCESSING_COUNT = 5000 # how many kmers of bv, tara, neotrop to compute
KMER_PROCESSING_STEPSIZE = 1000 # save file every KMER_PROCESSING_STEPSIZE
KMER_PROCESSING_VERBOSE = 10
INCUDE_TARA_BV_NEO = False
# -------------------- leave one out --------------------  #
REESTIMATE_TREE = True
REESTIMATE_TREE_SEQ_THRESHOLD = 30 # If REESTIMATE_TREE = True trees up to this no. of sequences will be reestimated
SEQUENCE_COUNT_THRESHOLD = 500 # dont consider too large alignments
SEQUENCE_LEN_THRESHOLD = 5000 # dont consider too long alignments