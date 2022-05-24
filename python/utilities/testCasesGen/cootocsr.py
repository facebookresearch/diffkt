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
import inverse

def gen(p, q, nonzerosRatio, permute):

    A_adj = utils.genRandomFloatSparseMatrix(p, q, nonzerosRatio)
    A = csr_matrix(A_adj)
    A_coo = coo_matrix(A)
    if permute:
        A_coo = utils.permuteCOOMatrix(A_coo)
        A = utils.orderPerservingCOOToCSR(A_coo)

    return A, A_coo

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='generate random sparse matrices, A with given non-zero probability, \
            then convert coo to csr')
    parse.add_argument('--rows', '-r', type=int, default=3, help='the number of rows for the input matrix A')
    parse.add_argument('--cols', '-c', type=int, default=3, help='the number of columns for the input matrix A')
    parse.add_argument('--nonzerosRatio', '-n', type=float, default=0.5, help='the ratio of non-zeros for A')
    parse.add_argument('--randomseed', '-s', type=int, default=0, help='the seed for numpy to generate random data')
    parse.add_argument('--permute', '-p', default=False, action='store_true', help='random shuffle the coo and then generate a order preserving CSR')
    args = parse.parse_args()

    np.random.seed(args.randomseed)

    A, A_coo = gen(args.rows, args.cols, args.nonzerosRatio, args.permute)

    print("----CPP format----")
    utils.printCOOMatrixCPP(A_coo, 'COO Input', '')
    utils.printCSRMatrixCPP(A, 'CSR Expected', 'E')
    print("----CPP format----")
