# -------------------- Feature config --------------------  #
K_MER_LENGTH = 15
K_MER_MAX_GAP_PERCENTAGE = 1.0
K_MER_MAX_N_PERCENTAGE = 1.0
BLOOM_FILTER_FP_RATE = 0.01
KMER_PROCESSING_INTERVAL_START = 0
KMER_PROCESSING_COUNT = 5000 # how many kmers of bv, tara, neotrop to compute
KMER_PROCESSING_STEPSIZE = 1000 # save file every KMER_PROCESSING_STEPSIZE
KMER_PROCESSING_VERBOSE = 10
INCUDE_TARA_BV_NEO = False
SIGN_ONLY_MATRIX_SIZE = 16
# -------------------- leave one out --------------------  #
LOO_SAMPLE_SIZE = 40
REESTIMATE_TREE = False
REESTIMATE_MSA = False
SKIP_EXISTING_PLACEMENTS_LOO = False
SEQUENCE_COUNT_THRESHOLD = 1000 # dont consider too large alignments
SEQUENCE_LEN_THRESHOLD = 10000 # dont consider too long alignments