from math import exp as exp
from math import floor as floor
from numpy import zeros as zeros

from copy import copy as copy

class BinaryMatrix:

    def __init__(self, matrix, rows, cols):
        """
        This class contains the algorithm specified in the NIST suite for computing the **binary rank** of a matrix.
        :param matrix: the matrix we want to compute the rank for
        :param rows: the number of rows
        :param cols: the number of columns
        :return: a BinaryMatrix object
        """
        self.M = rows
        self.Q = cols
        self.A = matrix
        self.m = min(rows, cols)

    def compute_rank(self, verbose=False):
        """
        This method computes the binary rank of self.matrix
        :param verbose: if this is true it prints out the matrix after the forward elimination and backward elimination
        operations on the rows. This was used to testing the method to check it is working as expected.
        :return: the rank of the matrix.
        """
        if verbose:
            print("Original Matrix\n", self.A)

        i = 0
        while i < self.m - 1:
            if self.A[i][i] == 1:
                self.perform_row_operations(i, True)
            else:
                found = self.find_unit_element_swap(i, True)
                if found == 1:
                    self.perform_row_operations(i, True)
            i += 1

        if verbose:
            print("Intermediate Matrix\n", self.A)

        i = self.m - 1
        while i > 0:
            if self.A[i][i] == 1:
                self.perform_row_operations(i, False)
            else:
                if self.find_unit_element_swap(i, False) == 1:
                    self.perform_row_operations(i, False)
            i -= 1

        if verbose:
            print("Final Matrix\n", self.A)

        return self.determine_rank()

    def perform_row_operations(self, i, forward_elimination):
        """
        This method performs the elementary row operations. This involves xor'ing up to two rows together depending on
        whether or not certain elements in the matrix contain 1's if the "current" element does not.
        :param i: the current index we are are looking at
        :param forward_elimination: True or False.
        """
        if forward_elimination:
            j = i + 1
            while j < self.M:
                if self.A[j][i] == 1:
                    self.A[j, :] = (self.A[j, :] + self.A[i, :]) % 2
                j += 1
        else:
            j = i - 1
            while j >= 0:
                if self.A[j][i] == 1:
                    self.A[j, :] = (self.A[j, :] + self.A[i, :]) % 2
                j -= 1

    def find_unit_element_swap(self, i, forward_elimination):
        """
        This given an index which does not contain a 1 this searches through the rows below the index to see which rows
        contain 1's, if they do then they swapped. This is done on the forward and backward elimination
        :param i: the current index we are looking at
        :param forward_elimination: True or False.
        """
        row_op = 0
        if forward_elimination:
            index = i + 1
            while index < self.M and self.A[index][i] == 0:
                index += 1
            if index < self.M:
                row_op = self.swap_rows(i, index)
        else:
            index = i - 1
            while index >= 0 and self.A[index][i] == 0:
                index -= 1
            if index >= 0:
                row_op = self.swap_rows(i, index)
        return row_op

    def swap_rows(self, i, ix):
        """
        This method just swaps two rows in a matrix. Had to use the copy package to ensure no memory leakage
        :param i: the first row we want to swap and
        :param ix: the row we want to swap it with
        :return: 1
        """
        temp = copy(self.A[i, :])
        self.A[i, :] = self.A[ix, :]
        self.A[ix, :] = temp
        return 1

    def determine_rank(self):
        """
        This method determines the rank of the transformed matrix
        :return: the rank of the transformed matrix
        """
        rank = self.m
        i = 0
        while i < self.M:
            all_zeros = 1
            for j in range(self.Q):
                if self.A[i][j] == 1:
                    all_zeros = 0
            if all_zeros == 1:
                rank -= 1
            i += 1
        return rank

class Matrix:

    @staticmethod
    def binary_matrix_rank_text(binary_data:str, verbose=False, rows_in_matrix = 32, columns_in_matrix = 32):
        """
        Note that this description is taken from the NIST documentation [1]
        [1] http://csrc.nist.gov/publications/nistpubs/800-22-rev1a/SP800-22rev1a.pdf
        The focus of the test is the rank of disjoint sub-matrices of the entire sequence. The purpose of this test is
        to check for linear dependence among fixed length sub strings of the original sequence. Note that this test
        also appears in the DIEHARD battery of tests.

        :param      binary_data         The seuqnce of bit being tested
        :param      verbose             True to display the debug messgae, False to turn off debug message
        :param      rows_in_matrix      Fixed for 32
        :param      columns_in_matrix   Fixed for 32
        :return     (p_value, bool)     A tuple which contain the p_value and result of frequency_test(True or False)
        """

        shape = (rows_in_matrix, columns_in_matrix)
        length_of_binary_data = len(binary_data)
        block_size = int(rows_in_matrix * columns_in_matrix)
        number_of_block = floor(length_of_binary_data / block_size)
        block_start = 0
        block_end = block_size

        if number_of_block > 0:
            max_ranks = [0, 0, 0]

            for im in range(number_of_block):
                block_data = binary_data[block_start:block_end]
                block = zeros(len(block_data))

                for count in range(len(block_data)):
                    if block_data[count] == '1':
                        block[count] = 1.0

                matrix = block.reshape(shape)
                ranker = BinaryMatrix(matrix, rows_in_matrix, columns_in_matrix)
                rank = ranker.compute_rank()

                if rank == rows_in_matrix:
                    max_ranks[0] += 1
                elif rank == (rows_in_matrix - 1):
                    max_ranks[1] += 1
                else:
                    max_ranks[2] += 1

                block_start += block_size
                block_end += block_size

            pi = [1.0, 0.0, 0.0]
            for x in range(1, 50):
                pi[0] *= 1 - (1.0 / (2 ** x))
            pi[1] = 2 * pi[0]
            pi[2] = 1 - pi[0] - pi[1]

            xObs = 0.0
            for i in range(len(pi)):
                xObs += pow((max_ranks[i] - pi[i] * number_of_block), 2.0) / (pi[i] * number_of_block)

            p_value = exp(-xObs / 2)

            if verbose:
                print('Binary Matrix Rank Test DEBUG BEGIN:')
                print("\tLength of input:\t", length_of_binary_data)
                print("\tSize of Row:\t\t", rows_in_matrix)
                print("\tSize of Column:\t\t", columns_in_matrix)
                print('\tValue of N:\t\t\t', number_of_block)
                print('\tValue of Pi:\t\t', pi)
                print('\tValue of xObs:\t\t', xObs)
                print('\tP-Value:\t\t\t', p_value)
                print('DEBUG END.')

            return (p_value, (p_value >= 0.01))
        else:
            return (-1.0, False)