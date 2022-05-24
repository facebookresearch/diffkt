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

def gen(p, q, nonzerosRatio):

    A_adj = utils.genRandomFloatSparseMatrix(p, q, nonzerosRatio)
    A = csr_matrix(A_adj)

    A_adj_T = A_adj.transpose()
    AT = csr_matrix(A_adj_T)

    return A, AT

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='generate random sparse matrices, A with given non-zero probability, \
            then transpose the matrix')
    parse.add_argument('--rows', '-r', type=int, default=3, help='the number of rows for the input matrix A')
    parse.add_argument('--cols', '-c', type=int, default=3, help='the number of columns for the input matrix A')
    parse.add_argument('--nonzerosRatio', '-n', type=float, default=0.5, help='the ratio of non-zeros for A')
    parse.add_argument('--randomseed', '-s', type=int, default=0, help='the seed for numpy to generate random data')
    args = parse.parse_args()

    np.random.seed(args.randomseed)

    A, AT = gen(args.rows, args.cols, args.nonzerosRatio)

    print("----CPP format----")
    utils.printCSRMatrixCPP(A, 'CSR Input', '')
    utils.printCSRMatrixCPP(AT, 'CSR Expected', 'E')
    print("----CPP format----")
