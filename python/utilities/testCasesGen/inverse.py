# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from scipy.sparse import csr_matrix
from numpy.linalg import inv

import argparse
import utils

def inverse(A):
    try:
        A_adj_inv = inv(A.todense())
        A_inv = csr_matrix(A_adj_inv)
    except np.linalg.LinAlgError:
        A_inv = csr_matrix((A.shape[1], A.shape[0]))
        print("A is not inversable")

    csr_matrix.sort_indices(A_inv)

    return A_inv

def gen(rows, cols, nonzerosRatio):
    A = utils.genRandomFloatSparseMatrix(rows, cols, nonzerosRatio)

    return A, inverse(A_inv)


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='generate random sparse matrices with probability, then perform inverse and print the input and the output')
    parse.add_argument('--rows', '-r', type=int, default=5, help='the number of rows for the input matrix')
    parse.add_argument('--cols', '-c', type=int, default=3, help='the number of columns for the input matrix')
    parse.add_argument('--nonzerosRatio', '-z', type=float, default=0.5, help='the ratio of non-zeros')
    parse.add_argument('--randomseed', '-s', type=int, default=0, help='the seed for numpy to generate random data')
    args = parse.parse_args()

    np.random.seed(args.randomseed)

    A, A_inv = gen(args.rows, args.cols, args.nonzerosRatio)

    utils.printCSRMatrix(A, 'input', '')
    utils.printCSRMatrix(A_inv, 'expected output', 'Inv')
