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
from inverse import inverse

def matdiv(A, B):
    B_inv = inverse(B)
    AdivB = A.dot(B_inv)
    csr_matrix.sort_indices(AdivB)

    return AdivB

def gen(p, k, q, nonzerosRatioA, nonzerosRatioB):

    B = utils.genRandomFloatSparseMatrix(k, q, nonzerosRatioB)
    A = utils.genRandomFloatSparseMatrix(p, k, nonzerosRatioA)

    return A, B, matdiv(A, B)

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='generate random sparse matrices, A, B with given non-zero probability, \
            then perform C =A/B (A: p by k, B: k by q, C: p by q)')
    parse.add_argument('--rowsA', '-p', type=int, default=3, help='the number of rows for the input matrix A')
    parse.add_argument('--colsA', '-k', type=int, default=3, help='the number of columns for the input matrix A')
    parse.add_argument('--colsB', '-q', type=int, default=3, help='the number of columns for the input matrix B')
    parse.add_argument('--nonzerosRatios', '-z', nargs="+", type=float, default=[0.5, 0.5], help='the ratios of non-zeros for A, B')
    parse.add_argument('--randomseed', '-s', type=int, default=0, help='the seed for numpy to generate random data')
    args = parse.parse_args()

    np.random.seed(args.randomseed)

    nonzerosRatioA, nonzerosRatioB = utils.processListTwo(args.nonzerosRatios, 0.5)
    A, B, AdivB = gen(args.rowsA, args.colsA, args.colsB, nonzerosRatioA, nonzerosRatioB)

    print("----CPP format----")
    utils.printCSRMatrixCPP(A, 'input A', 'A')
    utils.printCSRMatrixCPP(B, 'input B', 'B')
    utils.printCSRMatrixCPP(AdivB, 'expected output', 'Div')
    print("----CPP format----")

    print("----Kotlin format----")
    utils.printCOOMatrixKotlin(coo_matrix(A), 'input left', 't1')
    utils.printCOOMatrixKotlin(coo_matrix(B), 'input right', 't2')
    utils.printCOOMatrixKotlin(coo_matrix(AdivB), 'expected output', 'outExp')
    print("----Kotlin format----")
