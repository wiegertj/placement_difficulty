import os
import unittest

from Bio import SeqIO

from features import kmer_processing
from features.kmer_processing import multiprocess_string_kernel, filter_gapped_kmers, bloom_filter


class kmer_test(unittest.TestCase):
    def test_kmer_statistics(self):

        kmer_processing.main()




        self.assertEqual(True, False)



if __name__ == '__main__':
    unittest.main()
