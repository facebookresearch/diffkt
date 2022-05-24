# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from numpy.linalg import inv

import argparse
import utils

def gen(rows, cols, nonzerosRatio, permute):
    A_adj = utils.genRandomFloatSparseMatrix(rows, cols, nonzerosRatio)
    A_coo = coo_matrix(A_adj)
    if permute:
        A_coo = utils.permuteCOOMatrix(A_coo)

    A_csr = utils.orderPerservingCOOToCSR(A_coo)

    return A_coo, A_csr


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='generate random sparse matrices with given non-zero probability, generate the COO (permuted or not) and CSR format, and then print the output format')
    parse.add_argument('--rows', '-r', type=int, default=5, help='the number of rows for the input matrix')
    parse.add_argument('--cols', '-c', type=int, default=3, help='the number of columns for the input matrix')
    parse.add_argument('--nonzerosRatio', '-z', type=float, default=0.5, help='the ratio of non-zeros')
    parse.add_argument('--randomseed', '-s', type=int, default=0, help='the seed for numpy to generate random data')
    parse.add_argument('--permute', '-p', action='store_true', help='whether permute the COO or not')
    args = parse.parse_args()

    np.random.seed(args.randomseed)

    A_coo, A_csr = gen(args.rows, args.cols, args.nonzerosRatio, args.permute)

    utils.printCOOMatrixKotlin(A_coo, 'input', 't1')
    utils.printCSRMatrixKotlin(A_csr, 'expected output', 'Exp')
