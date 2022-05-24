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


def gen(b, p, k, q, nonzerosRatiosA, nonzerosRatiosB):

    As = []
    Bs = []
    Cs = []
    for i in range(b):
        A = coo_matrix(utils.genRandomFloatSparseMatrix(p, k, nonzerosRatiosA[i]))
        B = coo_matrix(utils.genRandomFloatSparseMatrix(k, q, nonzerosRatiosB[i]))
        C = coo_matrix(A.dot(B))
        As.append(A)
        Bs.append(B)
        Cs.append(C)

    return As, Bs, Cs

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='generate random sparse matrices, a batch of sparse matrices \
            A[0..b-1], B[0..b-1] with given non-zero probability, \
            then perform C[i]=A[i]/B[i] (A[i]: p by k, B[i]: k by q, C[i]: p by q) for i in [0..b-1]')
    parse.add_argument('--batchsize', '-b', type=int, default=2, help='the number of rows for the input matrix A')
    parse.add_argument('--rowsA', '-p', type=int, default=3, help='the number of rows for the input matrix A')
    parse.add_argument('--colsA', '-k', type=int, default=3, help='the number of columns for the input matrix A')
    parse.add_argument('--colsB', '-q', type=int, default=3, help='the number of columns for the input matrix B')
    parse.add_argument('--nonzerosRatios', '-z', nargs="+", type=float, default=[0.2], help='the ratios \
           of non-zeros for A[0..b-1], B[0..b-1]')
    parse.add_argument('--randomseed', '-s', type=int, default=0, help='the seed for numpy to generate random data')
    args = parse.parse_args()

    np.random.seed(args.randomseed)

    nonzerosRatiosA, nonzerosRatiosB = utils.processList(args.nonzerosRatios, args.batchsize)
    As, Bs, Cs = gen(args.batchsize, args.rowsA, args.colsA, args.colsB, nonzerosRatiosA, nonzerosRatiosB)

    print("----Kotlin format----")
    utils.printCOO3DMatrixKotlin(As, 'input left', 't1')
    utils.printCOO3DMatrixKotlin(Bs, 'input right', 't2')
    utils.printCOO3DMatrixKotlin(Cs, 'expected output', 'outExp')
    print("----Kotlin format----")
